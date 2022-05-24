from yacs.config import CfgNode as CN

_CN = CN()
_CN.OUTPUT = ''

# OETR Pipeline
_CN.OETR = CN()
_CN.OETR.CHECKPOINT = None
_CN.OETR.BACKBONE_TYPE = 'ResNet'
_CN.OETR.MODEL = 'oetr'  # options:['oetr', 'oetr_fc', 'oetr_fcos']
_CN.OETR.NORM_INPUT = True

# 1. OETR-backbone (local feature CNN) config
_CN.OETR.BACKBONE = CN()
_CN.OETR.BACKBONE.NUM_LAYERS = 50
_CN.OETR.BACKBONE.STRIDE = 16
_CN.OETR.BACKBONE.LAYER = 'layer3'  # options: ['layer4']
_CN.OETR.BACKBONE.LAST_LAYER = 1024  # output last channel size

# 2. OETR-neck module config
_CN.OETR.NECK = CN()
_CN.OETR.NECK.D_MODEL = 256
_CN.OETR.NECK.LAYER_NAMES = ['self', 'cross'] * 4
_CN.OETR.NECK.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.OETR.NECK.MAX_SHAPE = (
    100,
    100,
)  # max feature map shape, with image shape: max_shape*stride

# 3. OETR-neck module config
_CN.OETR.HEAD = CN()
_CN.OETR.HEAD.D_MODEL = 256
_CN.OETR.HEAD.NORM_REG_TARGETS = True

# 4. OETR-fine module config
_CN.OETR.LOSS = CN()
_CN.OETR.LOSS.OIOU = False
_CN.OETR.LOSS.CYCLE_OVERLAP = False
_CN.OETR.LOSS.FOCAL_ALPHA = 0.25
_CN.OETR.LOSS.FOCAL_GAMMA = 2.0
_CN.OETR.LOSS.REG_WEIGHT = 1.0
_CN.OETR.LOSS.CENTERNESS_WEIGHT = 1.0

# Dataset
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_ROOT = None

# training and validating
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.DATA_SOURCE = 'megadepth'
_CN.DATASET.TRAIN.LIST_PATH = 'assets/train_scenes.txt'
_CN.DATASET.TRAIN.PAIRS_LENGTH = None
_CN.DATASET.TRAIN.WITH_MASK = None
_CN.DATASET.TRAIN.TRAIN = True
_CN.DATASET.TRAIN.VIZ = False
_CN.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
_CN.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]

_CN.DATASET.VAL = CN()
_CN.DATASET.VAL.DATA_SOURCE = 'megadepth'
_CN.DATASET.VAL.LIST_PATH = 'assets/val_scenes.txt'
_CN.DATASET.VAL.PAIRS_LENGTH = None
_CN.DATASET.VAL.WITH_MASK = False
_CN.DATASET.VAL.OIOU = True
_CN.DATASET.VAL.TRAIN = False
_CN.DATASET.VAL.VIZ = True
_CN.DATASET.VAL.IMAGE_SIZE = [640, 640]
_CN.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
