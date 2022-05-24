#! /usr/bin/env python3

import argparse
import logging
import pprint
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n2048-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 2048,
            'keypoint_threshold':0.005
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'context-desc': {
        'output': 'feats-context-desc',
        'model': {
            'name': 'contextdesc',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'r2d2-desc': {
        'output': 'feats-r2d2-desc',
        'model': {
            'name': 'r2d2',
            'reliability_thr': 0.7,
            'repeatability_thr': 0.7,
            'topk': 5000
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'aslfeat-desc': {
        'output': 'feats-aslfeat-desc',
        'model': {
            'name': 'aslfeat',
            'max_keypoints': 2048,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'disk-desc': {
        'output': 'feats-disk-desc',
        'model': {
            'name': 'disk',
            'topk': 512,
            'weights': 'disk_depth.pth'
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
    'swin-disk-desc': {
        'output': 'feats-swin-disk-desc',
        'model': {
            'name': 'disk',
            'topk': 2048,
            'weights': 'disk-swin-conv.pth',
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
    'landmark': {
        'output': 'feats-landmark-desc',
        'model': {
            'name': 'landmark',
            'topk': 2048,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
}


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/'+g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = sorted(list(set(self.paths)))
        self.paths = [i.relative_to(root) for i in self.paths]
        logging.info(f'Found {len(self.paths)} images in root {root}.')

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and max(w, h) > self.conf.resize_max:
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': path.as_posix(),
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)

# import h5py
# @torch.no_grad()
# def main(conf, image_dir, export_dir, as_half=False, save_file=False):
#     logging.info('Extracting local features with configuration:'
#                  f'\n{pprint.pformat(conf)}')

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     Model = dynamic_load(extractors, conf['model']['name'])
#     model = Model(conf['model']).eval().to(device)

#     loader = ImageDataset(image_dir, conf['preprocessing'])
#     loader = torch.utils.data.DataLoader(loader, num_workers=1)
#     if save_file:
#         feature_path = Path(export_dir, conf['output']+'.h5')
#         feature_path.parent.mkdir(exist_ok=True, parents=True)
#         feature_file = h5py.File(str(feature_path), 'a')

#     for data in tqdm(loader):
#         if 'context' in conf['model']['name']:
#             pred = model(data)
#         else:
#             pred = model(map_tensor(data, lambda x: x.to(device)))
#         pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

#         pred['image_size'] = original_size = data['original_size'][0].numpy()
#         if 'keypoints' in pred:
#             size = np.array(data['image'].shape[-2:][::-1])
#             scales = (original_size / size).astype(np.float32)
#             pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

#         if as_half:
#             for k in pred:
#                 dt = pred[k].dtype
#                 if (dt == np.float32) and (dt != np.float16):
#                     pred[k] = pred[k].astype(np.float16)
#         if save_file:
#             grp = feature_file.create_group(data['name'][0])
#             for k, v in pred.items():
#                 grp.create_dataset(k, data=v)

#             del pred
#     if save_file:
#         feature_file.close()
#     logging.info('Finished exporting features.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir)
