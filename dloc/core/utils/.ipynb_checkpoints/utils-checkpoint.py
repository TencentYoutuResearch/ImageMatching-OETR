# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from threading import Thread

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    # if max(w_new, h_new) < 160:
    #     print('Warning: input resolution is very small, results may vary')
    # elif max(w_new, h_new) > 2000:
    #     print('Warning: input resolution is very large, results may vary')
    # from IPython import embed;embed()
    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_overlap_image(path, device, resize, rotation, resize_float, 
                grayscale=False, align='disk', overlap=False):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if not align or not overlap:
        image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None
    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    # w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    if align == 'disk':
        w_new = math.ceil(w_new/16)*16
        h_new = math.ceil(h_new/16)*16
    elif align == 'loftr':
        w_new = math.ceil(w_new/8)*8
        h_new = math.ceil(h_new/8)*8
        

    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    if overlap:
        overlap_inp = cv2.resize(image, (640, 640)).astype('float32')
        overlap_inp = overlap_inp[None]  # HxWxC to CxHxW

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inp = image[None, None]
    else:
        inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(inp/255.).float().to(device)
    overlap_inp = torch.from_numpy(overlap_inp/255.).float().to(device)
    return image, overlap_inp, inp, scales

def resize_pad_images(path, device, scale, rotation, resize_float, 
                grayscale=False, size_divisor=32, overlap=False):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img_color = image
    if not overlap:
        image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None

    h, w, c = image.shape
    scale_factor = min(scale[0]/w, scale[1]/h)
    new_size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
    # img_resize = cv2.resize(image, new_size)
    if resize_float:
        img_resize = cv2.resize(image.astype('float32'), new_size)
    else:
        img_resize = cv2.resize(image, new_size).astype('float32')

    #pad image
    pad_scale = [math.ceil(scale[0]/size_divisor)*size_divisor, 
                math.ceil(scale[1]/size_divisor)*size_divisor]

    overlap_inp = np.zeros((pad_scale[1], pad_scale[0], c), dtype=image.dtype)
    overlap_inp[:img_resize.shape[0], :img_resize.shape[1], :] = img_resize
    overlap_inp = overlap_inp[None]
    mask = np.zeros((int(pad_scale[1]/size_divisor), int(pad_scale[0]/size_divisor)), dtype=bool)
    mask[:int(img_resize.shape[0]/size_divisor), :int(img_resize.shape[1]/size_divisor)] = True
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inp = image[None, None]
    else:
        inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(inp/255.).float().to(device)
    overlap_inp = torch.from_numpy(overlap_inp/255.).float().to(device)
    mask = torch.from_numpy(mask)[None].float().to(device)
    return image, overlap_inp, inp, (1/scale_factor, 1/scale_factor), mask#, img_color

def dynamic_roi(img):
    xy = np.where(img > 40)
    if xy[0].shape[0] == 0:
        return [0, 0, img.shape[1], img.shape[0]], [0, 0, img.shape[1], img.shape[0]]
    roi = [xy[1].min(), xy[0].min(), xy[1].max(), xy[0].max()]
    offset_roi = [max(roi[0]-200, 0), max(roi[1]-200, 0), 
           min(roi[2]+200, img.shape[1]), min(roi[3]+200, img.shape[0])]
    return roi, offset_roi

def read_dynamic_mask_mat(image, device, resize, resize_float, 
                grayscale=False, align='disk', overlap=False, 
                rect=[], rect_flag=False):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None, None
    if rect:
        image = image[rect[1]:rect[3], rect[0]:rect[2]]
        _, d_rect = dynamic_roi(image)
        image = image[d_rect[1]:d_rect[3], d_rect[0]:d_rect[2]]
        d_rect = [d_rect[0] + rect[0], d_rect[1] + rect[1], d_rect[2] + rect[0], d_rect[3] + rect[1]]

    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    w_new, h_new = process_resize(w, h, resize)

    if align == 'disk':
        w_new = math.ceil(w_new/16)*16
        h_new = math.ceil(h_new/16)*16
    elif align == 'loftr':
        w_new = math.ceil(w_new/8)*8
        h_new = math.ceil(h_new/8)*8
    if w_new < 1 or h_new < 1:
        print('Failed size with :{:4}, {:4}'.format(w_new, h_new))
        exit(1)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if overlap:
        inp = image[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if grayscale:
            inp = image[None, None]
        else:
            inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not rect_flag:
        inp = torch.from_numpy(inp/255.).float().to(device)

    return image, inp, scales, d_rect

def read_image(path, device, resize, rotation, resize_float, 
                grayscale=False, align='disk', overlap=False):

    if grayscale:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None
    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    # w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    if align == 'disk':
        w_new = math.ceil(w_new/16)*16
        h_new = math.ceil(h_new/16)*16
    elif align == 'loftr':
        w_new = math.ceil(w_new/8)*8
        h_new = math.ceil(h_new/8)*8

    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    if overlap:
        inp = image[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if grayscale:
            inp = image[None, None]
        else:
            inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(inp/255.).float().to(device)

    return image, inp, scales

def overlap_crop(image1, bbox1, image2, bbox2):
    left = image1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    w1, h1 = left.shape[:2][::-1]
    w2, h2 = right.shape[:2][::-1]
    ratio1, ratio2 = (1, 1), (1, 1)
    if h1 > h2:
        w_new = int(float(h1)/float(h2)*w2)
        right = cv2.resize(right.astype('float32'), (w_new, h1))
        ratio2 = float(h1)/float(h2)
    else:
        w_new = int(float(h2)/float(h1)*w1)
        left = cv2.resize(left.astype('float32'), (w_new, h2))
        ratio1 = float(h2)/float(h1)
    return left, right, ratio1, ratio2

def patch_resize(origin_w, origin_h, w, h):
    if float(origin_w) / float(w) > float(origin_h) / float(h):
        ratio = float(origin_h) / float(h)
        new_w, new_h = ratio * float(w), origin_h
    else:
        ratio = float(origin_w) / float(w)
        new_w, new_h = origin_w, ratio * float(h)
    return ratio, int(new_w), int(new_h)

def overlap_filter(mkpts1, bbox1, mkpts2, bbox2):
    valid1 = np.logical_and(
            np.logical_and(mkpts1[:, 0] > bbox1[0], mkpts1[:, 0] < bbox1[2]),
            np.logical_and(mkpts1[:, 1] > bbox1[1], mkpts1[:, 1] < bbox1[3])
            )
    valid2 = np.logical_and(
            np.logical_and(mkpts2[:, 0] > bbox2[0], mkpts2[:, 0] < bbox2[2]),
            np.logical_and(mkpts2[:, 1] > bbox2[1], mkpts2[:, 1] < bbox2[3])
            )
    valid = np.logical_and(valid1, valid2)
    return valid

def tensor_overlap_crop(image1, bbox1, image2, bbox2):
    bbox1 = bbox1[0].int()
    bbox2 = bbox2[0].int()
    origin_w1, origin_h1 = image1.shape[2:][::-1]
    origin_w2, origin_h2 = image2.shape[2:][::-1]

    left = image1[0,:,bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[0,:,bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    
    w1, h1 = left.shape[1:][::-1]
    w2, h2 = right.shape[1:][::-1]
    if origin_w1*origin_h1 > origin_w2*origin_h2:
        ratio1, new_w1, new_h1 = patch_resize(origin_w1, origin_h1, w1, h1)
        ratio2, new_w2, new_h2 = patch_resize(origin_w1, origin_h1, w2, h2)
    else:
        ratio1, new_w1, new_h1 = patch_resize(origin_w2, origin_h2, w1, h1)
        ratio2, new_w2, new_h2 = patch_resize(origin_w2, origin_h2, w2, h2)

    cv_right = right.permute((1,2,0)).cpu().numpy()*255
    cv_left = left.permute((1,2,0)).cpu().numpy()*255
    # plot_image_pair([cv_left, cv_right])
    # plt.savefig('origin.jpg', bbox_inches='tight', pad_inches=0)
    # plt.close()

    cv_right = cv2.resize(cv_right.astype('float32'), (new_w2, new_h2), interpolation=cv2.INTER_CUBIC)
    cv_left = cv2.resize(cv_left.astype('float32'), (new_w1, new_h1), interpolation=cv2.INTER_CUBIC)
    if len(cv_right.shape) == 3:
        right = torch.from_numpy(cv_right/255).float().to(image1.device).permute((2,0,1))
        left = torch.from_numpy(cv_left/255).float().to(image1.device).permute((2,0,1))
    else:
        right = torch.from_numpy(cv_right/255).float().to(image1.device)[None]
        left = torch.from_numpy(cv_left/255).float().to(image1.device)[None]
    # cv2.imwrite('left.jpg', cv_left)
    # cv2.imwrite('right.jpg', cv_right)
    # plot_image_pair([cv_left, cv_right])
    # plt.savefig('test.jpg', bbox_inches='tight', pad_inches=0)
    # plt.close()
    return left[None], right[None], ratio1, ratio2

def visualize_overlap_crop(image1, bbox1, image2, bbox2, output=None):
    left = image1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    right = image2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    w1, h1 = left.shape[:2][::-1]
    w2, h2 = right.shape[:2][::-1]
    if h1 > h2:
        w_new = int(float(h1)/float(h2)*w2)
        right = cv2.resize(right.astype('float32'), (w_new, h1))
    else:
        w_new = int(float(h2)/float(h1)*w1)
        left = cv2.resize(left.astype('float32'), (w_new, h2))
    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right
    # viz = cv2.hconcat([left, right])
    # cv2.imwrite(output, viz)

def visualize_overlap(image1, bbox1, image2, bbox2, output=None):
    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]), (0, 0, 255), 7)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]), (0, 0, 255), 7)
    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right
    # viz = cv2.hconcat([left, right])
    # cv2.imwrite(output, viz)

def visualize_overlap_gt(image1, bbox1, gt1, image2, bbox2, gt2, output=None):
    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]), (255, 0, 0), 5)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]), (255, 0, 0), 5)
    left = cv2.rectangle(left, tuple(gt1[0:2]), tuple(gt1[2:]), (0, 255, 0), 5)
    right = cv2.rectangle(right, tuple(gt2[0:2]), tuple(gt2[2:]), (0, 255, 0), 5)

    if output:
        plot_image_pair([left, right])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return left, right

# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=20, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i].astype('uint8'), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        return make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 400,
                                opencv_display, opencv_title, small_text)
        # return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()

def keypoints_visualization(image, kpts):
    pass

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=100,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    # H, W = H0 + H1 + margin, max(W0, W1)
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    # out[H0+margin:, :W1, :] = image1

    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            # cv2.circle(out, (x, y + H0 + margin), 2, black, -1,
            #            lineType=cv2.LINE_AA)
            # cv2.circle(out, (x, y + H0 + margin), 1, white, -1,
            #            lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        # cv2.line(out, (x0, y0), (x1, y1 + H0 + margin),
        #          color=c, thickness=1, lineType=cv2.LINE_AA)
        # # display line end-points as circles
        # cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        # cv2.circle(out, (x1, y1 + H0 + margin), 2, c, -1,
        #            lineType=cv2.LINE_AA)
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

def save_h5(dict_to_save, filename):
    """Saves dictionary to hdf5 file"""
    import h5py
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])

def read_mask(json_file, scale, img_size):
        content = json.load(open(json_file, 'r', encoding='utf-8'))
        mask = np.zeros(img_size, np.uint8)
        for shape in content['shapes']:
            label = shape['label']
            roi = np.array(shape['points']).reshape((-1, 2)).astype(np.int32)
            roi = roi/scale
            if label == 'outer':
                cv2.rectangle(mask, roi, (255,))
            elif label == 'inner':
                cv2.rectangle(mask, roi, (0,))
        return mask

def read_outer_mask(json_file):
        content = json.load(open(json_file, 'r', encoding='utf-8'))
        w = int(content['imageWidth'])
        h = int(content['imageHeight'])
        mask = np.zeros((h, w), np.uint8)
        for shape in content['shapes']:
            label = shape['label']
            roi = np.array(shape['points']).reshape((-1, 2)).astype(np.int32)
            if label == 'outer':
                cv2.rectangle(mask, roi, (255,))
        return mask

def read_outer_rectangle(json_file, offset=200):
        content = json.load(open(json_file, 'r', encoding='utf-8'))
        w = int(content['imageWidth'])
        h = int(content['imageHeight'])
        mask = np.zeros((h, w), np.uint8)
        x0,y0 =10000, 10000
        x1,y1 =0, 0
        for shape in content['shapes']:
            label = shape['label']
            roi = np.array(shape['points']).reshape((-1, 2)).astype(np.int32)
            if label == 'outer' or label == 'object':
                x0 = min(x0,  max(roi[:,0].min() - offset, 0))
                y0 = min(y0,  max(roi[:,1].min() - offset, 0))
                x1 = max(x1, min(roi[:, 0].max() + offset, w))
                y1 = max(y1, min(roi[:, 1].max() + offset,h))
                
        # cv2.rectangle(mask, (x0, y0), (x1, y1), (255,))
        mask[y0:y1, x0:x1]=255
        return mask, [x0, y0, x1, y1], [max(x0 - 100, 0), max(y0 - 100, 0), min(x1 + 100, w), min(y1 + 100, h)]

def read_mask_mat(image, device, resize, resize_float, 
                grayscale=False, align='disk', overlap=False, rect=[]):

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)
    else:
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None
    if rect:
        # cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]),(0, 0, 255), 5)
        image = image[rect[1]:rect[3], rect[0]:rect[2]]
        # cv2.imwrite('outputs/template/{}'.format(path.split('/')[-1]), image)

    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    # w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)

    if align == 'disk':
        w_new = math.ceil(w_new/16)*16
        h_new = math.ceil(h_new/16)*16
    elif align == 'loftr':
        w_new = math.ceil(w_new/8)*8
        h_new = math.ceil(h_new/8)*8
    if w_new < 1 or h_new < 1:
        print('Failed size with :{:4}, {:4}'.format(w_new, h_new))
        exit(1)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if overlap:
        inp = image[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if grayscale:
            inp = image[None, None]
        else:
            inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(inp/255.).float().to(device)

    return image, inp, scales

def read_mask_image(path, device, resize, resize_float, 
                grayscale=False, align='disk', overlap=False, rect=[]):

    if grayscale:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if not align:
            image = image[:, :, ::-1]  # BGR to RGB
    if image is None:
        return None, None, None
    if rect:
        # cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]),(0, 0, 255), 5)
        image = image[rect[1]:rect[3], rect[0]:rect[2]]
        # cv2.imwrite('outputs/template/{}'.format(path.split('/')[-1]), image)

    image = image.astype(np.float32)
    w, h = image.shape[:2][::-1]
    # w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)

    if align == 'disk':
        w_new = math.ceil(w_new/16)*16
        h_new = math.ceil(h_new/16)*16
    elif align == 'loftr':
        w_new = math.ceil(w_new/8)*8
        h_new = math.ceil(h_new/8)*8
    if w_new < 1 or h_new < 1:
        from IPython import embed;embed()
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if overlap:
        inp = image[None]  # HxWxC to CxHxW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        if grayscale:
            inp = image[None, None]
        else:
            inp = image.transpose((2, 0, 1))[None]  # HxWxC to CxHxW
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(inp/255.).float().to(device)

    return image, inp, scales

def mask_filter(img, kpts):
    _,mask = cv2.threshold(img.astype('uint8'), 30, 255, cv2.THRESH_BINARY)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    mask = cv2.dilate(mask, verticalStructure)
    valid = mask[ kpts[:, 1].astype('int'),kpts[:, 0].astype('int')] == 255

    return kpts, mask, valid

def write_template():
    pass

def load_template(npz_path, device):
    template_info = np.load(npz_path, allow_pickle=True)
    keypoints = template_info['keypoints'].item()
    descriptors = template_info['descriptors'].item()
    scores = template_info['scores'].item()
    rois = template_info['rois'].item()
    pos = template_info['template_id'].item()
    scales = template_info['template_scale'].item()
    masks = template_info['template_mask'].item()
    images = template_info['template_img'].item()

    for k in keypoints.keys():
        keypoints[k] = torch.from_numpy(keypoints[k]).to(device)[None].float()
        descriptors[k] = torch.from_numpy(descriptors[k]).to(device)[None].float()
        scores[k] = torch.from_numpy(scores[k]).to(device)[None].float()
    return keypoints, descriptors, scores, scales, rois, pos, masks, images

