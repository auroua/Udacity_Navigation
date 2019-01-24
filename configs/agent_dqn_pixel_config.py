# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.HYPER_PARAMETER = CN()
# Replay Memory Size
_C.HYPER_PARAMETER.BUFFER_SIZE = int(1e7)
# Batch Size
_C.HYPER_PARAMETER.BATCH_SIZE = 256
# Soft Update of Target Parameters
_C.HYPER_PARAMETER.TAU = 1e-3
# Learning Rate
_C.HYPER_PARAMETER.LR = 0.00025
# MOMENTUM
_C.HYPER_PARAMETER.MOMENTUM = 0.95
# NetWork Update Frequency
_C.HYPER_PARAMETER.UPDATE_EVERY = 4
# Target Network Update Frequency
_C.HYPER_PARAMETER.TARGET_EVERY = 5000
# Total Training Episodes
_C.HYPER_PARAMETER.EPISODES = 200000
# eps-greedy eps start value
_C.HYPER_PARAMETER.EPS_START = 1.0
# eps-greedy eps end value
_C.HYPER_PARAMETER.EPS_END = 0.01
# eps-greedy eps decay rate
_C.HYPER_PARAMETER.EPS_DECAY = 0.995
# Discount Factor
_C.HYPER_PARAMETER.GAMMA = 0.99
# Define the Loss Type. Support ('MSE', 'F1')
_C.HYPER_PARAMETER.LOSS_TYPE = 'MSE'
# The Model Name Used to Approximate Q-Function  ('QNetwork',)
_C.HYPER_PARAMETER.MODEL_TYPE = 'QPixelNetwork'
# The Training Agent Type
_C.HYPER_PARAMETER.AGENT_TYPE = 'DqnPixelAgent'
# The Optimizer used for training (SGD, ADAM)
_C.HYPER_PARAMETER.OPTIMIZER = 'SGD'
# Training algorithms
_C.HYPER_PARAMETER.INTERFACE = 'dqn_pixel'

_C.MODEL_PARAMETER = CN()
# Fully Connection Model Hidden Layer Parameter
_C.MODEL_PARAMETER.H1 = 512
_C.MODEL_PARAMETER.H2 = 512
_C.MODEL_PARAMETER.H3 = 512


def get_dqn_pix_cfg_defaults():
    return _C.clone()
