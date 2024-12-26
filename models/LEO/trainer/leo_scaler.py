import os
import random
from collections import defaultdict
from datetime import timedelta
from math import ceil

import torch
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import (InitProcessGroupKwargs, ProjectConfiguration,
                              set_seed)
from common.misc import CustomAccelerator, default_collate, split_train_set
from data.build import build_dataloader_leo, get_dataset_leo
from model.leo_agent import LeoAgent
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange
from trainer.build import (TRAINER_REGISTRY, Tracker, build_optim,
                           latest_checkpoint)
from trainer.leo_trainer import LeoTrainer

logger = get_logger(__name__)


@TRAINER_REGISTRY.register()
class LeoScaler(LeoTrainer):

    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.epochs = cfg.training.epochs

        # initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        gradient_accumulation_steps = cfg.training.get(
            'gradient_accumulation_steps', 1)

        self.accelerator = CustomAccelerator(
            project_config=ProjectConfiguration(
                project_dir=self.exp_dir,
                automatic_checkpoint_naming=True,
                total_limit=1,
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs)

        # dataset, dataloader, evaluator
        self.eai_task_sources = ['hm3d', 'mp3d', 'cliport']
        self.data_loaders = {'train': []}  # list of subsets
        train_set = get_dataset_leo(
            cfg=cfg,
            split='train',
            dataset_name=cfg.task.leomix.dataset,
            dataset_wrapper_name=cfg.task.leomix.dataset_wrapper,
            dataset_wrapper_args=cfg.task.leomix.dataset_wrapper_args)
        train_subsets = split_train_set(train_set, self.epochs)
        for train_subset in train_subsets:
            self.data_loaders['train'].append(
                DataLoader(
                    train_subset,
                    batch_size=cfg.dataloader.train.batchsize,
                    num_workers=cfg.dataloader.train.num_workers,
                    collate_fn=getattr(train_subset, 'collate_fn',
                                       default_collate),
                    pin_memory=True,
                    shuffle=True,
                    drop_last=True,
                ))

        self.data_loaders['val'] = build_dataloader_leo(
            cfg=cfg,
            split='test',
            dataset_name=cfg.task.leomix.dataset,
            dataset_wrapper_name=cfg.task.leomix.dataset_wrapper,
            dataset_wrapper_args=cfg.task.leomix.dataset_wrapper_args,
            dataloader_args=cfg.task.leomix.eval_dataloader_args,
        )

        # prepare dataloaders
        self.data_loaders['train'] = [
            self.accelerator.prepare(sub_loader)
            for sub_loader in self.data_loaders['train']
        ]
        self.data_loaders['val'] = self.accelerator.prepare(
            self.data_loaders['val'])

        # build model
        self.model = LeoAgent(cfg)
        learnable_named_params = self.model.get_learnable_named_params()
        self.accelerator.learn_params_list = list(
            learnable_named_params.keys())
        optim_params = list(learnable_named_params.values())

        # prepare model, optimizer and scheduler
        total_steps = sum([
            ceil(len(sub_loader) / gradient_accumulation_steps)
            for sub_loader in self.data_loaders['train']
        ])
        self.optimizer, self.scheduler = build_optim(cfg,
                                                     optim_params,
                                                     total_steps=total_steps)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler)

        self.exp_tracker = Tracker(cfg)
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # load checkpoints
        resume_ckpt = latest_checkpoint(
            os.path.join(self.exp_dir, 'checkpoints'))

        if resume_ckpt:
            load_model_only = False
            self.pretrained_ckpt_path = resume_ckpt
            logger.info(
                f'Train: resume and load state from {self.pretrained_ckpt_path}'
            )
        elif cfg.pretrained_ckpt_path and os.path.exists(
                cfg.pretrained_ckpt_path):
            load_model_only = True
            self.pretrained_ckpt_path = cfg.pretrained_ckpt_path
            logger.info(
                f'Train: start and load model from {self.pretrained_ckpt_path}'
            )
        else:
            self.pretrained_ckpt_path = None
            logger.info('Train: start from scratch')

        if self.pretrained_ckpt_path is not None:
            self.load(path=self.pretrained_ckpt_path,
                      model_only=load_model_only)

        # misc
        self.grad_norm = cfg.training.grad_norm

        self.accelerator.init_trackers(
            project_name=cfg.name,
            config=OmegaConf.to_container(cfg,
                                          resolve=True,
                                          throw_on_missing=True),
            init_kwargs={
                'wandb': {
                    'name': self.exp_tracker.exp_name,
                    'entity': cfg.logger.entity,
                    'id': self.exp_tracker.run_id,
                    'resume': True
                }
            })

    def train_step(self, epoch):
        logger.info(f'Start training epoch {epoch+1}')
        self.model.train()
        loader = self.data_loaders['train'][
            epoch]  # the only difference to LeoTrainer.train_step()
        pbar = trange(len(loader),
                      disable=(not self.accelerator.is_main_process))

        if self.exp_tracker.loader_step > 0:
            logger.info(
                f'Skip the first {self.exp_tracker.loader_step} batches')
            loader = self.accelerator.skip_first_batches(
                loader, self.exp_tracker.loader_step)
            pbar.update(self.exp_tracker.loader_step)

        for data_dict in loader:
            with self.accelerator.accumulate(self.model):
                # categorize tasks
                is_txt_data = [(s not in self.eai_task_sources)
                               for s in data_dict['source']]
                is_eai_data = [(s in self.eai_task_sources)
                               for s in data_dict['source']]

                # forward
                data_dict = self.forward(data_dict, inference=False)

                # calculate loss and optimize
                loss = data_dict['loss']
                loss_all = loss.mean()
                self.backward(loss_all)

                # record
                loss_dict = {'overall': loss_all}
                loss_txt = loss[is_txt_data]
                loss_eai = loss[is_eai_data]
                if len(loss_txt) > 0:
                    loss_dict.update({'txt': loss_txt.mean()})
                if len(loss_eai) > 0:
                    loss_dict.update({'eai': loss_eai.mean()})
                self.log(loss_dict, mode='train', task='loss')
                self.exp_tracker.step_loader()
                pbar.update(1)

        logger.info(f'Finish training epoch {epoch+1}')

    @torch.no_grad()
    def val_step(self, epoch):
        logger.info(f'Start validation epoch {epoch+1}')
        self.model.eval()
        loader = self.data_loaders['val']
        pbar = trange(len(loader),
                      disable=(not self.accelerator.is_main_process))
        all_losses = defaultdict(list)
        for data_dict in loader:
            # convert list to str for training forward
            if not isinstance(data_dict['output_gt'][0], str):
                data_dict['output_gt'] = [
                    random.choice(answer_list)
                    for answer_list in data_dict['output_gt']
                ]

            # inference
            data_dict = self.forward(data_dict, inference=False)

            # gather
            data_dict_non_tensor = {
                k: v
                for k, v in data_dict.items()
                if not isinstance(v, torch.Tensor)
            }
            data_dict_non_tensor = self.accelerator.gather_for_metrics(
                data_dict_non_tensor)
            data_dict = {
                k: v
                for k, v in data_dict.items() if isinstance(v, torch.Tensor)
            }
            data_dict = self.accelerator.gather_for_metrics(data_dict)
            data_dict.update(data_dict_non_tensor)

            all_losses['overall'].append(data_dict['loss'])

            is_txt_data = [(s not in self.eai_task_sources)
                           for s in data_dict['source']]
            is_eai_data = [(s in self.eai_task_sources)
                           for s in data_dict['source']]
            loss_txt = data_dict['loss'][is_txt_data]
            loss_eai = data_dict['loss'][is_eai_data]
            if len(loss_txt) > 0:
                all_losses['txt'].append(loss_txt)
            if len(loss_eai) > 0:
                all_losses['eai'].append(loss_eai)

            pbar.update(1)

        loss_dict = {}
        for k, v in all_losses.items():
            loss_dict[k] = torch.cat(v).mean().item()

        self.log(loss_dict, mode='val', task='loss')
        logger.info(
            f'Finish validation epoch {epoch+1}, test set loss: {loss_dict}')

    def run(self):
        start_epoch = self.exp_tracker.epoch
        for epoch in range(start_epoch, self.epochs):
            self.train_step(epoch)
            self.exp_tracker.step()
            self.save(model_only=False)  # automatic checkpointing
            self.accelerator.wait_for_everyone()
            self.val_step(epoch)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
