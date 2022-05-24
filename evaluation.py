#!/usr/bin/env python
"""
@File    :   main.py
@Time    :   2021/10/14 15:56:00
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib import cm as cm
from tqdm import tqdm

from dloc.api import extract_process
from dloc.core import (extract_features, extractors, match_features, matchers,
                       overlap_features, overlaps)
from dloc.core.match_features import preprocess_match_pipeline
from dloc.core.overlap_features import preprocess_overlap_pipeline
from dloc.core.utils.base_model import dynamic_load
from dloc.core.utils.utils import (make_matching_plot, tensor_overlap_crop,
                                   vis_aligned_image)

torch.set_grad_enabled(False)


class Matching(torch.nn.Module):
    """Image Matching Frontend (extractor + matcher)"""
    def __init__(self, config=None, model_path=Path('weights/')):
        super(Matching, self).__init__()
        # if config is None:
        #     self.config = {}
        # else:
        self.config = config

        if self.config['overlaper'] is not None:
            self.overlap = dynamic_load(overlaps,
                                        config['overlaper']['model']['name'])(
                                            config['overlaper']['model'],
                                            model_path)
        if not self.config['direct']:
            self.extractor = dynamic_load(
                extractors, config['extractor']['model']['name'])(
                    config['extractor']['model'], model_path)
        self.matcher = dynamic_load(matchers,
                                    config['matcher']['model']['name'])(
                                        config['matcher']['model'], model_path)
        self.extractor_name = self.config['extractor']['model']['name']
        self.matcher_name = self.config['matcher']['model']['name']
        self.size_divisor = 1

    def forward(self, data, with_overlap=False):
        """Run extractors and matchers
        Extractor is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        if self.config['direct'] and not with_overlap:
            return self.matcher(data)

        if with_overlap:
            pred = {}
            self.device = data['overlap_image1'].device
            assert isinstance(data['overlap_scales0'], tuple)
            assert isinstance(data['overlap_scales1'], tuple)
            data['overlap_scales0'] = torch.tensor(data['overlap_scales0'] +
                                                   data['overlap_scales0'],
                                                   device=self.device)
            data['overlap_scales1'] = torch.tensor(data['overlap_scales1'] +
                                                   data['overlap_scales1'],
                                                   device=self.device)
            bbox0, bbox1 = self.overlap({
                'image0': data['overlap_image0'],
                'image1': data['overlap_image1']
            })

            bbox0 = bbox0 * data['overlap_scales0']
            bbox1 = bbox1 * data['overlap_scales1']

            bw0, bh0 = (
                bbox0[0][2].int() - bbox0[0][0].int(),
                bbox0[0][3].int() - bbox0[0][1].int(),
            )
            bw1, bh1 = (
                bbox1[0][2].int() - bbox1[0][0].int(),
                bbox1[0][3].int() - bbox1[0][1].int(),
            )
            overlap_scores = max(
                torch.floor_divide(bw0, bw1),
                torch.floor_divide(bh0, bh1),
                torch.floor_divide(bw1, bw0),
                torch.floor_divide(bh1, bh0),
            )

            if min(bw0, bh0, bw1, bh1) > 1 and (
                (data['dataset_name'] == 'pragueparks-val'
                 and overlap_scores > 2.0)
                    or data['dataset_name'] != 'pragueparks-val'):
                if self.matcher_name == 'loftr':
                    self.size_divisor = 8
                overlap0, overlap1, ratio0, ratio1 = tensor_overlap_crop(
                    data['image0'],
                    bbox0,
                    data['image1'],
                    bbox1,
                    self.extractor_name,
                    self.size_divisor,
                )

                pred.update({
                    'bbox0':
                    bbox0,
                    'bbox1':
                    bbox1,
                    'ratio0':
                    torch.tensor(ratio0, device=bbox0.device),
                    'ratio1':
                    torch.tensor(ratio1, device=bbox0.device),
                })
                if self.config['direct']:
                    matches = self.matcher({
                        'image0': overlap0,
                        'image1': overlap1
                    })
                    pred.update(matches)
                    return pred
                # Extract SuperPoint (keypoints, scores, descriptors) if not provided
                if 'keypoints0' not in data:
                    # self.extractor.net.config['nms_radius'] = 3 * math.ceil(ratio0)
                    pred0 = self.extractor({'image': overlap0})
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    # self.extractor.net.config['nms_radius'] = 3 * math.ceil(ratio1)
                    pred1 = self.extractor({'image': overlap1})
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))

            else:
                pred.update({
                    'bbox0':
                    torch.tensor(
                        [[
                            0.0,
                            0.0,
                            data['image0'].shape[3],
                            data['image0'].shape[2],
                        ]],
                        device=bbox0.device,
                    ),
                    'bbox1':
                    torch.tensor(
                        [[
                            0.0,
                            0.0,
                            data['image1'].shape[3],
                            data['image1'].shape[2],
                        ]],
                        device=bbox0.device,
                    ),
                    'ratio0':
                    torch.tensor([[1.0, 1.0]], device=bbox0.device),
                    'ratio1':
                    torch.tensor([[1.0, 1.0]], device=bbox0.device),
                })
                if self.config['direct']:
                    matches = self.matcher(data)
                    pred.update(matches)
                    return pred
                if 'keypoints0' not in data:
                    pred0 = self.extractor({'image': data['image0']})
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    pred1 = self.extractor({'image': data['image1']})
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))

        else:
            if not self.config['direct']:
                pred = extract_process(self.extractor, data)
                pred.update({
                    'bbox0':
                    torch.tensor(
                        [[
                            0.0,
                            0.0,
                            data['image0'].shape[3],
                            data['image0'].shape[2],
                        ]],
                        device=data['image0'].device,
                    ),
                    'bbox1':
                    torch.tensor(
                        [[
                            0.0,
                            0.0,
                            data['image1'].shape[3],
                            data['image1'].shape[2],
                        ]],
                        device=data['image0'].device,
                    ),
                    'ratio0':
                    torch.tensor([[1.0, 1.0]], device=data['image0'].device),
                    'ratio1':
                    torch.tensor([[1.0, 1.0]], device=data['image0'].device),
                })

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data.update(pred)
        for k in data:
            if isinstance(data[k], (list, tuple)) and k not in [
                    'overlap_scales0',
                    'overlap_scales1',
            ]:
                data[k] = torch.stack(data[k])

        # Perform the matching
        matches = self.matcher(data)
        pred.update(matches)
        return pred


def save_h5(dict_to_save, filename):
    """Saves dictionary to hdf5 file."""

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def viz_pairs(output, image0, image1, name0, name1, mconf, kpts0, kpts1,
              mkpts0, mkpts1):
    """visualization image pairs."""
    viz_path = output + '{}_{}_matches.png'.format(
        name0.split('/')[-1],
        name1.split('/')[-1])
    # Visualize the matches.
    color = cm.jet(mconf)
    text = [
        'Matcher',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    small_text = [
        'Image Pair: {}:{}'.format(name0.split('/')[-1],
                                   name1.split('/')[-1]),
    ]

    make_matching_plot(
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        color,
        text,
        viz_path,
        True,
        True,
        False,
        'Matches',
        small_text,
    )


def main(
    config,
    input,
    input_pairs,
    output,
    with_desc=False,
    resize=None,
    resize_float=False,  # True,
    viz=False,
    save=False,
    evaluate=False,
    warp_origin=True,
):
    """main pipeline of matching process."""
    if resize is None:
        resize = [-1]
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matching = Matching(config).eval().to(device)
    if not os.path.exists(output):
        os.makedirs(output)

    with open(input_pairs, 'r') as f:
        pairs = [line.split() for line in f.readlines()]

    seq_keypoints = defaultdict(dict)
    seq_descriptors = defaultdict(dict)
    seq_matches = defaultdict(dict)
    seq_pose = defaultdict(dict)
    seq_inparams = defaultdict(dict)
    seq_scale = defaultdict(dict)
    for i, pair in tqdm(enumerate(pairs), total=len(pairs)):
        name0, name1 = pair[:2]

        # if "googleurban-val" not in name0:
        #     continue
        gray = config['extractor']['preprocessing']['grayscale']
        # Load the image pair.
        align = ''
        if 'disk' in config['extractor']['output']:
            align = 'disk'
        elif 'loftr' in config['matcher']['output']:
            align = 'loftr'
            with_desc = False
        if 'megadepth' in input_pairs:
            scene = name0.split('/')[1]
        elif 'imc' in input_pairs:
            scene = name0.split('/')[0] + '/' + name0.split('/')[1]
        elif 'fuchi' in input_pairs:
            scene = name0.split('/')[-1][:-4]
        else:
            scene = name0.split('/')[0]

        if config['overlaper'] is not None:
            results = preprocess_overlap_pipeline(
                input,
                name0,
                name1,
                device,
                resize,
                resize_float,
                gray,
                align,
                config,
                pair,
                matching,
                with_desc,
                warp_origin=warp_origin,
            )
        else:
            results = preprocess_match_pipeline(
                input,
                name0,
                name1,
                device,
                resize,
                resize_float,
                gray,
                align,
                config,
                pair,
                matching,
                with_desc,
                warp_origin=warp_origin,
            )
            if 'icp' in config['matcher']['model']['name']:
                viz_path = name0 + '_' + name1
                if viz and i % 10 == 0:
                    vis_aligned_image(results['mask0'], results['mask1'],
                                      results['T_0_1'], viz_path)
                seq_pose[scene]['{}-{}'.format(
                    name0.split('/')[-1][:-4],
                    name1.split('/')[-1][:-4])] = results['T_0_1']
                continue

        # Write matching results
        if 'loftr' in config['matcher']['output'] or config[
                'overlaper'] is not None:
            # For all matcher-dependent keypoints
            im0, im1 = name0.split('/')[-1][:-4], name1.split('/')[-1][:-4]
            if '{}-{}'.format(im0, im1) not in seq_keypoints[scene]:
                seq_keypoints[scene]['{}-{}'.format(im0,
                                                    im1)] = results['kpts0']
                if config['overlaper'] is not None and not warp_origin:
                    seq_inparams[scene]['{}-{}'.format(
                        im0, im1)] = np.concatenate(
                            (
                                np.array(results['scales0']),
                                results['oxy0'],
                                results['ratio0'],
                            ),
                            axis=-1,
                        )
                if with_desc:
                    seq_descriptors[scene]['{}-{}'.format(
                        im0, im1)] = results['desc0']
            if '{}-{}'.format(im1, im0) not in seq_keypoints[scene]:
                seq_keypoints[scene]['{}-{}'.format(im1,
                                                    im0)] = results['kpts1']
                if config['overlaper'] is not None and not warp_origin:
                    seq_inparams[scene]['{}-{}'.format(
                        im1, im0)] = np.concatenate(
                            (
                                np.array(results['scales1']),
                                results['oxy1'],
                                results['ratio1'],
                            ),
                            axis=-1,
                        )
                if with_desc:
                    seq_descriptors[scene]['{}-{}'.format(
                        im1, im0)] = results['desc1']
        else:
            if name0 not in seq_keypoints[scene]:
                seq_keypoints[scene][name0.split('/')[-1]
                                     [:-4]] = results['kpts0']
                if with_desc:
                    seq_descriptors[scene][name0.split('/')[-1]
                                           [:-4]] = results['desc0']

            if name1 not in seq_keypoints[scene]:
                seq_keypoints[scene][name1.split('/')[-1]
                                     [:-4]] = results['kpts1']
                if with_desc:
                    seq_descriptors[scene][name1.split('/')[-1]
                                           [:-4]] = results['desc1']
        seq_matches[scene]['{}-{}'.format(
            name0.split('/')[-1][:-4],
            name1.split('/')[-1][:-4])] = np.concatenate([[results['index0']],
                                                          [results['index1']]])
        # get scale different between image pairs
        if 'ratio0' in results and 'ratio1' in results:
            seq_scale[scene]['{}-{}'.format(
                name0.split('/')[-1][:-4],
                name1.split('/')[-1][:-4])] = min(
                    results['ratio0'][0] / results['ratio1'][0],
                    results['ratio1'][0] / results['ratio0'][0],
                )

        if viz and i % 10 == 0:
            if not os.path.exists(os.path.join(output, 'viz')):
                os.makedirs(os.path.join(output, 'viz'))

            viz_pairs(
                os.path.join(output, 'viz/{}_'.format(scene.replace('/',
                                                                    '-'))),
                results['image0'],
                results['image1'],
                name0,
                name1,
                results['mconf'],
                results['kpts0'],
                results['kpts1'],
                results['kpts0'][results['index0']],
                results['kpts1'][results['index1']],
            )
    # Save results to h5 file
    if save:
        for k in seq_keypoints.keys():
            if not os.path.exists(os.path.join(output, k)):
                os.makedirs(os.path.join(output, k))
            if with_desc:
                save_h5(seq_descriptors[k],
                        os.path.join(output, k, 'descriptors.h5'))
            save_h5(seq_keypoints[k], os.path.join(output, k, 'keypoints.h5'))
            save_h5(seq_matches[k], os.path.join(output, k, 'matches.h5'))
            if len(seq_inparams) > 0:
                save_h5(seq_inparams[k], os.path.join(output, k,
                                                      'inparams.h5'))
            if len(seq_scale) > 0:
                save_h5(seq_scale[k], os.path.join(output, k, 'scales.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/pairs.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='assets/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
        'the visualization images are written',
    )

    parser.add_argument(
        '--matcher',
        choices={
            'superglue_outdoor',
            'superglue_disk',
            'superglue_swin_disk',
            'superglue_indoor',
            'NN',
            'disk',
            'cotr',
            'loftr',
        },
        default='indoor',
        help='SuperGlue weights',
    )
    parser.add_argument(
        '--extractor',
        choices={
            'superpoint_aachen',
            'superpoint_inloc',
            'd2net-ss',
            'r2d2-desc',
            'context-desc',
            'landmark',
            'aslfeat-desc',
            'disk-desc',
            'swin-disk-desc',
        },
        default='superpoint_aachen',
        help='SuperGlue weights',
    )
    parser.add_argument(
        '--overlaper',
        choices={
            'oetr',
            'oetr_imc',
        },
        default=None,
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs='+',
        default=[-1],
        help='Resize the input image before running inference. If two numbers, '
        'resize to the exact dimensions, if one number, resize the max '
        'dimension, if -1, do not resize',
    )
    parser.add_argument('--with_desc',
                        action='store_true',
                        help='Saving without descriptors')
    parser.add_argument('--viz',
                        action='store_true',
                        help='Visualization matching results')
    parser.add_argument('--landmark',
                        action='store_true',
                        help='Keypoints extraction with landmarks')
    parser.add_argument(
        '--direct',
        action='store_true',
        help='Match images directe without keypoints extraction.',
    )
    parser.add_argument('--save',
                        action='store_true',
                        help='save match results')
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='validation results online')
    parser.add_argument(
        '--warp_origin',
        action='store_false',
        help='Warp keypoints to origin image scale.',
    )
    opt = parser.parse_args()
    extractor_conf = extract_features.confs[opt.extractor]
    matcher_conf = match_features.confs[opt.matcher]
    if opt.overlaper is None:
        overlaper_conf = None
    else:
        overlaper_conf = overlap_features.confs[opt.overlaper]
    config = {
        'landmark': opt.landmark,
        'extractor': extractor_conf,
        'matcher': matcher_conf,
        'direct': opt.direct,
        'overlaper': overlaper_conf,
    }
    if overlaper_conf is not None:
        output_path = os.path.join(
            opt.output_dir,
            opt.extractor + '_' + opt.matcher + '_' + opt.overlaper + '/',
        )
    else:
        output_path = os.path.join(opt.output_dir,
                                   opt.extractor + '_' + opt.matcher + '/')
    main(
        config,
        opt.input_dir,
        opt.input_pairs,
        output_path,
        opt.with_desc,
        opt.resize,
        viz=opt.viz,
        save=opt.save,
        evaluate=opt.evaluate,
        warp_origin=opt.warp_origin,
    )
