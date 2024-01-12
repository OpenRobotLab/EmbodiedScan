from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.sampler_seed_hook import DistSamplerSeedHook
from mmengine.runner.log_processor import LogProcessor

default_scope = 'embodiedscan'

default_hooks = dict(timer=dict(type=IterTimerHook),
                     logger=dict(type=LoggerHook, interval=50),
                     param_scheduler=dict(type=ParamSchedulerHook),
                     checkpoint=dict(type=CheckpointHook,
                                     interval=1,
                                     max_keep_ckpts=1),
                     sampler_seed=dict(type=DistSamplerSeedHook))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
