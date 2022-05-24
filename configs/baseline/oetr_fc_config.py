from src.config.default import _CN as cfg

cfg.OUTPUT = 'oetr_fc'

cfg.DATASET.DATA_ROOT = './dataset/megadepth/'

cfg.OETR.MODEL = 'oetr_fc'  # options:['sacdnet', 'safdnet']
cfg.OETR.NORM_INPUT = True
cfg.OETR.CHECKPOINT = None

# 1. OETR-backbone (local feature CNN) config
cfg.OETR.BACKBONE.NUM_LAYERS = 50
cfg.OETR.BACKBONE.STRIDE = 32
cfg.OETR.BACKBONE.LAYER = 'layer4'  # options: ['layer4']
cfg.OETR.BACKBONE.LAST_LAYER = 2048  # output last channel size

cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.TRAIN.LIST_PATH = './dataset/megadepth/assets/megadepth_train_pairs.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000

cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.VAL.LIST_PATH = './dataset/megadepth/assets/megadepth_validation_scale.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None

cfg.OETR.LOSS.OIOU = False
cfg.OETR.LOSS.CYCLE_OVERLAP = False
