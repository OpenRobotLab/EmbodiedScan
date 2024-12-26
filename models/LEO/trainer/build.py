import copy as cp
import math
import os
from datetime import datetime

import torch.optim as optim
import wandb
from common.type_utils import cfg2dict
from fvcore.common.registry import Registry
from torch.optim.lr_scheduler import LambdaLR

TRAINER_REGISTRY = Registry('Trainer')


class Tracker():

    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1
        self.loader_step = 0

    def step_loader(self):
        self.loader_step += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.note}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        self.run_id = wandb.util.generate_id()
        self.epoch = 0
        self.loader_step = 0
        self.overall_best_result = 0

    def state_dict(self):
        return {
            'run_id': self.run_id,
            'epoch': self.epoch,
            'exp_name': self.exp_name,
            'loader_step': self.loader_step,
            'overall_best_result': self.overall_best_result,
        }

    def load_state_dict(self, state_dict):
        state_dict = cp.deepcopy(state_dict)
        self.run_id = state_dict['run_id']
        self.epoch = state_dict['epoch']
        self.loader_step = state_dict['loader_step']
        self.exp_name = state_dict['exp_name']
        self.overall_best_result = state_dict['overall_best_result']


def linear_warmup_cosine_decay(step, warmup_step, total_step):
    if step <= warmup_step:
        return 1e-3 + step / warmup_step * (1 - 1e-3)
    return max(
        0.5 * (1 + math.cos(
            (step - warmup_step) / (total_step - warmup_step) * math.pi)),
        1e-5)


def get_scheduler(cfg, optimizer, total_steps):
    lambda_func = lambda step: globals()[cfg.training.schedule.name](
        step, cfg.training.schedule.args.warmup_steps, total_steps)
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)


def build_optim(cfg, params, total_steps):
    optimizer = getattr(optim, cfg.training.optim.name)(params, **cfg2dict(
        cfg.training.optim.args))
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    return optimizer, scheduler


def latest_checkpoint(path):
    if not os.path.exists(path):
        return ''
    checkpoints = [os.path.join(path, f) for f in os.listdir(path)]
    if len(checkpoints) == 0:
        return ''
    return max(checkpoints, key=os.path.getmtime)


def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
