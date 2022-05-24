import argparse
import glob
import math
import os

import cv2

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import random as rng
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage import measure, transform
from tqdm import tqdm

from .api import Extracting, build_model
from .core.utils.utils import (make_matching_plot, read_dynamic_mask_mat,
                               read_mask_image, read_mask_mat,
                               read_outer_rectangle)

torch.set_num_threads(1)
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


class IndurtailPipeline():
    def __init__(self,
                template_path='assets/template',
                extractor_mode='disk-desc',
                matcher_mode='superglue_jabil',
                dloc_path='.',
                device='cuda:0',
                resize=[612],
                roi_offset=200,
                similar_thresh=0.7,
                improved_thresh=0.45,
                rectangle_pos=['001', '003', '009', '011', 
                               '011B', '015', '015B', '017', 
                            '019', '047', '049', '051', 
                            '067', '079', '081', '083'],
                failed_pos=['226', '227', '236']):
        self.weight_path = os.path.join(dloc_path, 'weights')
        self.matcher, self.config = build_model(extractor_mode,
                                              matcher_mode,
                                              device,
                                              self.weight_path,
                                              landmark=False,
                                              direct=False,
                                              template=True)
        self.template_path = template_path
        self.device = device
        self.resize = resize
        self.roi_offset = roi_offset
        self.rectangle_pos=rectangle_pos
        self.failed_pos = failed_pos
        self.improved_thresh = improved_thresh
        self.gray = self.config['extractor']['preprocessing']['grayscale']
        self.align = ''
        if 'disk' in self.config['extractor']['output']:
            self.align = 'disk'
        elif 'loftr' in self.config['matcher']['output']:
            self.align = 'loftr'
        self.similar_thresh = similar_thresh
        if template_path:
            self.template_info = self.template_extract()
        else:
            self.template_info = self.template_load(os.path.join(dloc_path, 'template_cache/template.npz'))
        self.common_threshold = similar_thresh + 0.1
        self.curve_threshold = similar_thresh - 0.1
        
    def mask_filter(self, img, rect_flag=False):
        _,mask = cv2.threshold(img.astype('uint8'), 40, 255, cv2.THRESH_BINARY)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        if rect_flag:
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            mask = cv2.dilate(mask, verticalStructure)
        else:
            mask = cv2.dilate(mask, verticalStructure)
        return mask
    
    def get_template_info(self):
        return self.template_info
    
    def template_load(self, npz_path):
        template_info = np.load(npz_path, allow_pickle=True)['template_info'].item()
        for k in template_info['keypoint'].keys():
            template_info['keypoint'][k] = torch.from_numpy(template_info['keypoint'][k]
                                            ).to(self.device)[None]
            template_info['descriptor'][k] = torch.from_numpy(template_info['descriptor'][k]
                                            ).to(self.device)[None]
            template_info['score'][k] = torch.from_numpy(template_info['score'][k]
                                            ).to(self.device)[None]
        return template_info

    def template_extract(self):
        template_files = os.listdir(self.template_path)
        os.listdir(self.template_path)
        template_infos = defaultdict(dict)
        for f in tqdm(template_files, total=len(template_files)):
            if f[-4:] == 'json':
                continue
            pos = f.split('/')[-1][:-4]
            if pos[-1] not in ['B'] and int(pos) in [2, 4, 5, 6, 8, 10, 12, 13, 14, 16, 18, 20, 32, 34, 36,
                48, 50, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 68, 80, 82, 84]:
                continue
            direct = 'left'
            if pos[1:3] in ['09', '19', '67', '83', '61']:
                direct = 'right'

            tempalte_mask, roi, _ = read_outer_rectangle(os.path.join(self.template_path, pos+'.json'), offset=self.roi_offset)

            if (pos[-1] != 'B' and (int(pos) in curve_pos)) or (pos[-1] == 'B' and (int(pos[:-1]) in curve_pos)): 
                image = cv2.imread(os.path.join(self.template_path, pos+'.jpg'), -1)
                img, inp, scale, d_roi = read_dynamic_mask_mat(image, self.device, self.resize, 
                                                  True, self.gray, self.align, rect = roi)
            else:
                img, inp, scale = read_mask_image(os.path.join(self.template_path, pos+'.jpg'), self.device, self.resize, 
                                                  True, self.gray, self.align, rect = roi)
                d_roi = roi

            rect_flag = pos in self.rectangle_pos
            mask = self.mask_filter(img, rect_flag)
            pred = self.matcher.forward_extractor({'image': inp})
            kpts = pred['keypoints'][0].cpu().numpy()
            desc = pred['descriptors'][0].cpu().numpy()
            score = pred['scores'][0].cpu().numpy()
            template_infos['keypoint'][pos] = kpts
            template_infos['descriptor'][pos] = desc
            template_infos['score'][pos] = score
            template_infos['scale'][pos] = scale
            template_infos['mask'][pos] = mask
            template_infos['roi'][pos] = roi
            template_infos['droi'][pos] = d_roi
            if rect_flag:
                rect_info = self.common_rectangle(mask)
                template_infos['corners'][pos] = rect_info[0]
                template_infos['anchor'][pos] = self.sample_line_points((template_infos['corners'][pos][0] + template_infos['corners'][pos][1])/2,
                                                                (template_infos['corners'][pos][2] + template_infos['corners'][pos][3])/2,
                                                                rect_info[1], direct)
                template_infos['mask'][pos] = rect_info[1]

        return template_infos
    
    def pose_estimation(self, pred):
        kpts0 = pred['keypoints0']
        kpts1 = pred['keypoints1']
        matches = pred['matches0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        if mkpts0.shape[0] < 4:
            return np.eye(3), 0
        M, _ = cv2.estimateAffine2D(mkpts1, mkpts0, None, cv2.RANSAC, 5, maxIters=3000)
        M = np.concatenate((M[0][None], M[1][None], np.array([[0,0,1]])), axis=0)
        return M, 1

    def origin_pose(self, M, scale0, scale1, roi0, roi1):
        K0 = np.array([[scale0[0], 0, roi0[0]], 
                      [0, scale0[1], roi0[1]], 
                      [0,0,1]]
                      )
        K1 = np.array([[1./scale1[0], 0, -roi1[0]/scale1[0]], 
                       [0, 1./scale1[1], -roi1[1]/scale1[1]], 
                       [0,0,1]]
                       )
        M_origin = K0 @ M @ K1
        return M_origin
    
    def similarity_evaluation(self, roi_binary, cur_binary, M):
        xy = np.where(cur_binary==255)
        
        h_pos = torch.cat([torch.from_numpy(xy[1][None]), 
                           torch.from_numpy(xy[0][None]), 
                           torch.ones(xy[0].shape[0])[None]], 
                          dim=0).double()
        
        warp_h_pos = torch.matmul(torch.from_numpy(M).double(), h_pos)
        warp_h_pos = warp_h_pos[:-1, :]/warp_h_pos[-1, :]
        warp_h_pos = warp_h_pos.numpy()

        warp_valid = (warp_h_pos[1, :] < roi_binary.shape[0]) * (warp_h_pos[1, :] >= 0) \
                    * (warp_h_pos[0, :] < roi_binary.shape[1]) * (warp_h_pos[0, :] >= 0)
        warp_h_pos_valid = warp_h_pos[:, warp_valid].astype(int)
        h_pos_valid = h_pos[:, warp_valid].numpy().astype(int)
        warp_mask = np.zeros(roi_binary.shape)
        warp_mask[warp_h_pos_valid[1, :], warp_h_pos_valid[0, :]] = 1
        warp_num = warp_mask.sum()
        repeate_num = warp_h_pos_valid.shape[1] - warp_num

        binary_same = abs(cur_binary[h_pos_valid[1, :], h_pos_valid[0, :]] - \
                      roi_binary[warp_h_pos_valid[1, :], warp_h_pos_valid[0, :]]) < 10
        similar_coeff = (np.sum(binary_same) - repeate_num) / max(h_pos.shape[1], np.where(roi_binary==255)[0].shape[0])
        return similar_coeff
    
    def viz_resize_crop(self, image, mask, M, viz_path):
        perspective_img = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        max_shape = [max(perspective_img.shape[0], mask.shape[0]), max(perspective_img.shape[1], mask.shape[1])]
        pad_perspective = np.zeros((max_shape[0], max_shape[1]))
        pad_mask = np.zeros((max_shape[0], max_shape[1]))
        pad_perspective[:perspective_img.shape[0], :perspective_img.shape[1]] = perspective_img
        pad_mask[:mask.shape[0], :mask.shape[1]] = mask
        pad_mask = np.stack([pad_mask.astype('uint8')]*3, -1)
        pad_mask[:,:,1:2] = 0
        pad_perspective = cv2.addWeighted(np.stack([pad_perspective]* 3, -1), 0.8, pad_mask, 0.2, 0, dtype = cv2.CV_32F)
        pad_img = np.zeros((max_shape[0], max_shape[1]))
        pad_img[:image.shape[0], :image.shape[1]] = image
        origin_img = cv2.addWeighted(np.stack([pad_img]* 3, -1), 0.8, pad_mask, 0.2, 0, dtype = cv2.CV_32F)
        video_img = np.hstack((origin_img, pad_perspective))
        cv2.imwrite(viz_path, video_img)
        
    def check_AB_template(self, mask_a, mask_b, mask):
        mask_num = np.sum(mask == 255)
        if np.abs(np.sum(mask_a == 255) - mask_num) < np.abs(np.sum(mask_b == 255) - mask_num):
            return ''
        else:
            return 'B' 
    
    def sample_line_points(self, xy1, xy2, mask, direct='left'):
        if xy2[0] - xy1[0] < 1e-5:
            return xy1

        fuc_a = (xy2[1] - xy1[1])/(xy2[0] - xy1[0])
        detla = xy1[1] - xy1[0]*(xy2[1] - xy1[1])/(xy2[0] - xy1[0])
        x_cor = np.linspace(0, mask.shape[1], mask.shape[1]).astype(int)

        y_cor = (fuc_a * x_cor + detla).astype(int)
        valid = (y_cor < mask.shape[0]) * (x_cor < mask.shape[1]) * (y_cor >= 0) * (x_cor >= 0)
        y_cor = y_cor[valid]
        x_cor = x_cor[valid]
        eq_num = (mask[y_cor[:-1], x_cor[:-1]] == mask[y_cor[1:], x_cor[1:]])

        if np.sum(eq_num) == len(eq_num):
            if direct == 'left':
                return xy1
            else:
                return xy2
        else:
            index = np.where(eq_num == False)[0]
            if direct == 'left':
                return [x_cor[index[0]], y_cor[index[0]]]
            else:
                return [x_cor[index[-1]], y_cor[index[-1]]]

    def common_rectangle(self, mask, debug=False):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        minRect = []
        rectangle_mask = np.zeros(mask.shape)
        if debug:
            viz_img = np.stack([rectangle_mask]*3, -1)

        max_rect_id = -1
        max_rect_area = 0
        rect_list = []
        contours_list = []
        for _, c in enumerate(contours):
            min_rect = cv2.minAreaRect(c)
            if min_rect[1][0] * min_rect[1][1] < 5000:
                continue
            if min_rect[1][0] * min_rect[1][1] > max_rect_area:
                max_rect_area = min_rect[1][0] * min_rect[1][1]
                max_rect_id += 1
            if abs(min_rect[2] - 90) < 10:
                min_rect = (min_rect[0], (min_rect[1][1], min_rect[1][0]), 0)
            rect_list.append(min_rect)
            contours_list.append(c)

        for i, min_rect in enumerate(rect_list):
            if abs(min_rect[1][1]/rect_list[max_rect_id][1][1] - 1.0) > 0.1 or \
               abs(min_rect[0][1] - rect_list[max_rect_id][0][1]) > 0.1*rect_list[max_rect_id][0][1] :
                continue
            box = cv2.boxPoints(min_rect)
            minRect.append(box)
            cv2.fillPoly(rectangle_mask, [contours_list[i]], (255,))
            if debug:
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                cv2.drawContours(viz_img, contours, i, color)
                # rotated rectangle
                box_int = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.fillPoly(viz_img, [contours_list[i]], (255,255,255))
                cv2.drawContours(viz_img, [box_int], 0, color, 10)
                cv2.imwrite('mask.png', viz_img)
        
        if len(minRect) in [1, 2]:
            x_cor = [minRect[i][j][0] for i in range(len(minRect)) for j in range(4)]
            y_cor = [minRect[i][j][1] for i in range(len(minRect)) for j in range(4)]
            x_cor.sort()
            y_cor.sort()
            corners = np.array([np.array([x_cor[0], y_cor[0]]), np.array([x_cor[0], y_cor[-1]]), 
                                np.array([x_cor[-1], y_cor[-1]]), np.array([x_cor[-1], y_cor[0]])])
        else:
            return [np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32), mask, False]
        corners = corners.clip(0)
        return [corners, rectangle_mask, True]

    def rectangle_corners(self, mask):
        rect_thresh = mask.shape[0] * 122
        if np.sum(mask[:, 0]) < rect_thresh and np.sum(mask[:, mask.shape[1] - 1]) < rect_thresh:
            return [np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32), mask, False]
        if np.sum(mask[:, 0]) >= rect_thresh and np.sum(mask[:, mask.shape[1] - 1]) >= rect_thresh:
            left_xy = np.where(mask[:, 0] == 255)[0]
            right_xy = np.where(mask[:, mask.shape[1] - 1] == 255)[0]
            up_left, bottom_left = [0, left_xy.min()], [0, left_xy.max()]
            up_right, bottom_right = [mask.shape[1] - 1, right_xy.min()],\
                                    [mask.shape[1] - 1, right_xy.max()]
            corners = np.array([bottom_left, up_left, up_right, bottom_right], dtype=np.float32)
            return [corners, True]
        return self.common_rectangle(mask)
    
    def pose_check(self, M):
        sx = math.sqrt(M[0][0]*M[0][0]+M[0][1]*M[0][1])
        sy = math.sqrt(M[1][0]*M[1][0]+M[1][1]*M[1][1])
        theta_x = math.acos(M[0][0] / sx) * 180 / math.pi
        theta_y = math.acos(M[1][1] / sy) * 180 / math.pi
        if math.fabs(sx-1) > 0.1 or math.fabs(theta_x) > 10 or \
            math.fabs(sy-1) > 0.1 or math.fabs(theta_y) > 10:
            return False
        return True
        
    def process(self, image, pos='P001', debug=False, output='', img_file='', time_cost={}):
        """To be implemented by the child class."""
        raise NotImplementedError