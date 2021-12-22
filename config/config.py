# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 8

_C.TRAIN = CN()
_C.TRAIN.IMG_SIZE = 473
_C.TRAIN.BENCHMARK = 'pascal'
_C.TRAIN.BSZ = 20
_C.TRAIN.LR = 1e-4
_C.TRAIN.DECODER_LR = 1e-4
_C.TRAIN.LR_SCHEDULER = 'constant' # constant | cosine
_C.TRAIN.NITER = 1000
_C.TRAIN.FOLD = 0
_C.TRAIN.BACKBONE = 'resnet101'
_C.TRAIN.CATS_AUGMENTATIONS = False
_C.TRAIN.PFENET_AUGMENTATIONS = False
_C.TRAIN.MASK = False
_C.TRAIN.WEIGHT_DECAY = 0.
_C.TRAIN.DECODER_WEIGHT_DECAY = 0.05
_C.TRAIN.MILESTONES = [100, 150]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`