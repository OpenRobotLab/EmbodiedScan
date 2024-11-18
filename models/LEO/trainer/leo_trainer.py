import json
import os
from datetime import timedelta
from math import ceil

import torch
import torch.nn as nn
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import (InitProcessGroupKwargs, ProjectConfiguration,
                              set_seed)
from common.io_utils import make_dir
from common.misc import CustomAccelerator
from data.build import build_dataloader_leo
from evaluator.build import build_eval_leo
from model.leo_agent import LeoAgent
from omegaconf import OmegaConf
from tqdm import trange
from trainer.build import (TRAINER_REGISTRY, Tracker, build_optim,
                           latest_checkpoint)

logger = get_logger(__name__)

model_parallel_classes = (
    nn.parallel.DistributedDataParallel,
    nn.DataParallel,
)


@TRAINER_REGISTRY.register()
class LeoTrainer():

    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.mode = cfg.mode

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
        self.data_loaders = {'train': {}, 'val': {}, 'test': {}}
        self.evaluators = {}
        self.eval_metrics = {}
        for task_name in cfg.task.keys():
            if cfg.task[task_name] and 'dataset' in cfg.task[task_name]:
                for mode in cfg.task[task_name].mode:
                    self.data_loaders[mode][task_name] = build_dataloader_leo(
                        cfg=cfg,
                        split=mode,
                        dataset_name=cfg.task[task_name].dataset,
                        dataset_wrapper_name=cfg.task[task_name].
                        dataset_wrapper,
                        dataset_wrapper_args=cfg.task[task_name].
                        dataset_wrapper_args,
                        dataloader_args=cfg.task[task_name].
                        train_dataloader_args if mode == 'train' else
                        cfg.task[task_name].eval_dataloader_args,
                    )
                if 'evaluator' in cfg.task[task_name]:
                    self.evaluators[task_name] = build_eval_leo(
                        cfg, task_name, cfg.task[task_name].evaluator)
                    self.eval_metrics[task_name] = 0

        assert len(self.data_loaders['train']
                   ) <= 1, 'LEO requires only one training set'

        # prepare dataloaders
        all_loaders, all_loader_keys = [], []
        for mode, loaders in self.data_loaders.items():
            for task, loader in loaders.items():
                all_loader_keys.append((mode, task))
                all_loaders.append(loader)
        accelerate_loaders = self.accelerator.prepare(*all_loaders)
        for k, v in zip(all_loader_keys, accelerate_loaders):
            self.data_loaders[k[0]][k[1]] = v

        # build model
        self.model = LeoAgent(cfg)
        learnable_named_params = self.model.get_learnable_named_params()
        self.accelerator.learn_params_list = list(
            learnable_named_params.keys())
        optim_params = list(learnable_named_params.values())

        # prepare model, optimizer and scheduler
        total_steps = ceil(
            len(list(self.data_loaders['train'].values())[0]) /
            gradient_accumulation_steps) * cfg.task.training.epochs
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
        self_best_ckpt = os.path.join(self.exp_dir, 'best.pth')
        print(self.mode)
        if self.mode == 'train':
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
        else:
            if os.path.exists(self_best_ckpt):
                self.pretrained_ckpt_path = self_best_ckpt
            elif cfg.pretrained_ckpt_path and os.path.exists(
                    cfg.pretrained_ckpt_path):
                self.pretrained_ckpt_path = cfg.pretrained_ckpt_path
            else:
                raise ValueError('No checkpoint to load for evaluation')
            load_model_only = True
            logger.info(f'Eval: load model from {self.pretrained_ckpt_path}')

        if self.pretrained_ckpt_path is not None:
            self.load(path=self.pretrained_ckpt_path,
                      model_only=load_model_only)

        # misc
        self.epochs = cfg.training.epochs
        self.grad_norm = cfg.training.grad_norm
        self.val_interval = cfg.eval.val_interval
        self.num_batch_val = cfg.eval.num_batch_val

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

    def forward(self, data_dict, inference=False):
        if inference:
            if isinstance(self.model, model_parallel_classes):
                return self.model.module.generate(data_dict)
            else:
                return self.model.generate(data_dict)
        else:
            return self.model(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(),
                                             self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        logger.info(f'Start training epoch {epoch+1}')
        self.model.train()
        loader = list(self.data_loaders['train'].values())[0]
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
    def val_step(self, epoch, full_val=False):
        logger.info(f'Start validation epoch {epoch+1}')
        self.model.eval()
        for task_name in self.evaluators.keys():
            if task_name in self.data_loaders['val']:
                loader = self.data_loaders['val'][task_name]
                pbar = trange(len(loader),
                              disable=(not self.accelerator.is_main_process))
                for i, data_dict in enumerate(loader):

                    # inference
                    data_dict = self.forward(data_dict, inference=True)

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
                        for k, v in data_dict.items()
                        if isinstance(v, torch.Tensor)
                    }
                    data_dict = self.accelerator.gather_for_metrics(data_dict)
                    data_dict.update(data_dict_non_tensor)

                    self.evaluators[task_name].update(data_dict)
                    pbar.update(1)

                _, results = self.evaluators[task_name].record(
                    split='val',
                    is_main_process=self.accelerator.is_main_process)

                self.eval_metrics[task_name] = results['target_metric']
                self.log(results, mode='val', task=task_name)
                logger.info(f'{task_name}: {results}')
                self.evaluators[task_name].reset()

        # simply summing up
        overall_avg_metrics = sum(list(self.eval_metrics.values())) / len(
            self.eval_metrics)
        self.log({'avg_metrics': overall_avg_metrics},
                 mode='val',
                 task='overall')
        if overall_avg_metrics > self.exp_tracker.overall_best_result:
            is_best = True
            self.exp_tracker.overall_best_result = overall_avg_metrics
        else:
            is_best = False
        logger.info(f'Finish validation epoch {epoch+1}, is_best = {is_best}')
        return is_best

    @torch.no_grad()
    def test_step(self):
        logger.info('Start final testing')
        self.model.eval()
        for task_name in self.evaluators.keys():
            if task_name in self.data_loaders['test']:
                loader = self.data_loaders['test'][task_name]
                pbar = trange(len(loader),
                              disable=(not self.accelerator.is_main_process))
                for idx, data_dict in enumerate(loader):
                    data_dict = self.forward(data_dict, inference=True)

                    data_dict_non_tensor = {
                        k: v
                        for k, v in data_dict.items()
                        if not isinstance(v, torch.Tensor)
                    }
                    data_dict_non_tensor = self.accelerator.gather_for_metrics(
                        data_dict_non_tensor)
                    data_dict = {
                        k: v
                        for k, v in data_dict.items()
                        if isinstance(v, torch.Tensor)
                    }
                    data_dict = self.accelerator.gather_for_metrics(data_dict)
                    data_dict.update(data_dict_non_tensor)

                    self.evaluators[task_name].update(data_dict)
                    pbar.update(1)

                    if idx % 500 == 0:
                        json.dump(
                            self.evaluators[task_name].save_results,
                            open(
                                os.path.join(self.exp_dir,
                                             f'test_{task_name}_{idx}.json'),
                                'w'))
                json.dump(
                    self.evaluators[task_name].save_results,
                    open(
                        os.path.join(self.exp_dir,
                                     f'test_{task_name}_complete.json'), 'w'))

                _, results = self.evaluators[task_name].record(
                    split='test',
                    is_main_process=self.accelerator.is_main_process)

                self.log(results, mode='test', task=task_name)
                logger.info(f'{task_name}: {results}')
                self.evaluators[task_name].reset()

        logger.info('Finish testing')

    def log(self, results, mode='train', task='default'):
        log_dict = {}
        for key, val in results.items():
            log_dict[f'{mode}/{task}/{key}'] = val

        if mode == 'train':
            lrs = self.scheduler.get_lr()
            for i, lr in enumerate(lrs):
                log_dict[f'train/lr/group_{i}'] = lr

        self.accelerator.log(log_dict)

    def save(self, name='best.pth', model_only=False):
        if model_only:
            path = os.path.join(self.exp_dir, name)
            make_dir(path)
            model_state_dict = self.accelerator.get_state_dict(self.model)
            # automatically filter non-learnable params, and save on main_process
            self.accelerator.save(model_state_dict,
                                  os.path.join(path, 'pytorch_model.bin'))
        else:
            self.accelerator.save_state(
            )  # automatic_checkpoint_naming = True -> self.exp_dir / checkpoints

    def load(self, path, model_only=False):
        if model_only:
            if os.path.exists(os.path.join(path, 'pytorch_model.bin')):
                model_state_dict = torch.load(
                    os.path.join(path, 'pytorch_model.bin'))
            else:
                model_state_dict = torch.load(path)
            if isinstance(self.model, model_parallel_classes):
                self.model.module.load_state_dict(model_state_dict,
                                                  strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # resume training
            self.accelerator.load_state(path, strict=False)
            self.accelerator.project_configuration.iteration = int(
                str(path)[-1]) + 1
        logger.info(
            f'Successfully loaded from {str(path)}, load_model_only = {model_only}'
        )

    def run(self):
        if self.mode == 'train':
            start_epoch = self.exp_tracker.epoch
            for epoch in range(start_epoch, self.epochs):

                self.train_step(epoch)
                # if (epoch + 1) % self.val_interval == 0:
                #     is_best = self.val_step(epoch)

                #     if is_best:
                self.save('model_last.pth', model_only=True)
                self.accelerator.wait_for_everyone()

                self.exp_tracker.step()
                self.save(model_only=False)  # automatic checkpointing
                self.accelerator.wait_for_everyone()

            # load best checkpoint for test
            logger.info('Training finished, load best checkpoint for testing')
            self.load(os.path.join(self.exp_dir, 'best.pth'), model_only=True)

        self.test_step()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
