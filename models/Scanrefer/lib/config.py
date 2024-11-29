import os
import sys
from easydict import EasyDict

CONF = EasyDict()


ENV_PATH = os.path.abspath(__file__)
# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = os.path.dirname(os.path.dirname(ENV_PATH))
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "scanrefer_1031_0.2_25epoch") 

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126
CONF.TRAIN.SEED = 42
