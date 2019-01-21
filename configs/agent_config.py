# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.HYPER_PARAMETER = CN()
# Replay Memory Size
_C.HYPER_PARAMETER.BUFFER_SIZE = int(1e5)
# Batch Size
_C.HYPER_PARAMETER.BATCH_SIZE = 64
# Soft Update of Target Parameters
_C.HYPER_PARAMETER.TAU = 1e-3
# Learning Rate
_C.HYPER_PARAMETER.LR = 5e-4
# NetWork Update Frequency
_C.HYPER_PARAMETER.UPDATE_EVERY = 4
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.99
# Define the Loss Type. Support ('MSE', 'F1')
_C.HYPER_PARAMETER.LOSS_TYPE = 'F1'
# The Model Name Used to Approximate Q-Function  ('QNetwork',)
_C.HYPER_PARAMETER.MODEL_TYPE = 'QNetwork'

_C.ENV_PARAMETER = CN()
# ENV State Size
_C.ENV_PARAMETER.STATE_SIZE = 38
# ENV Action Size
_C.ENV_PARAMETER.ACTION_SIZE = 4

_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 128
_C.MODEL_PARAMETER.H2 = 256
_C.MODEL_PARAMETER.H3 = 128


def get_cfg_defaults():
    return _C.clone()
