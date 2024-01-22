from mmengine import DATASETS as MMENGINE_DATASETS
from mmengine import METRICS as MMENGINE_METRICS
from mmengine import MODELS as MMENGINE_MODELS
from mmengine import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine import Registry

MODELS = Registry('model',
                  parent=MMENGINE_MODELS,
                  locations=['embodiedscan.models'])
DATASETS = Registry('dataset',
                    parent=MMENGINE_DATASETS,
                    locations=['embodiedscan.datasets'])
TRANSFORMS = Registry('transform',
                      parent=MMENGINE_TRANSFORMS,
                      locations=['embodiedscan.datasets.transforms'])
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=['embodiedscan.eval'])
TASK_UTILS = Registry('task util',
                      parent=MMENGINE_TASK_UTILS,
                      locations=['embodiedscan.models'])
