import json
import os
from datetime import datetime
from math import ceil

import common.io_utils as iu
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from common.misc import rgetattr
from data.data_utils import pad_tensors
from data.datasets import LeoBase
from model.leo_agent import LeoAgent
from tqdm import trange
from trainer.leo_trainer import LeoTrainer

logger = get_logger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LeoProber(LeoTrainer, LeoBase):

    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.rscan_base = cfg.data.rscan_base
        self.scannet_base = cfg.data.scan_family_base
        self.num_points = cfg.data.num_points
        self.max_obj_len = cfg.data.max_obj_len
        self.batch_size = cfg.dataloader.eval.batchsize
        self.split = 'test'
        self.save_obj_tokens = cfg.probe.save_obj_tokens

        # dummpy accelerator
        self.accelerator = Accelerator()

        # load model
        self.model = LeoAgent(cfg)
        self.model.to(device)
        self.model.eval()

        self_best_ckpt = os.path.join(self.exp_dir, 'best.pth')
        if os.path.exists(self_best_ckpt):
            self.pretrained_ckpt_path = self_best_ckpt
        elif cfg.pretrained_ckpt_path and os.path.exists(
                cfg.pretrained_ckpt_path):
            self.pretrained_ckpt_path = cfg.pretrained_ckpt_path
        else:
            raise ValueError('No checkpoint to load for evaluation')

        logger.info(f'Probe: load model from {self.pretrained_ckpt_path}')
        self.load(path=self.pretrained_ckpt_path, model_only=True)

        # prepare data
        self.sources = [cfg.probe.sources] if isinstance(
            cfg.probe.sources, str) else list(cfg.probe.sources)
        self.scene_ids = [cfg.probe.scene_ids] if isinstance(
            cfg.probe.scene_ids, str) else list(cfg.probe.scene_ids)
        self.situations = [cfg.probe.situations] if isinstance(
            cfg.probe.situations, str) else list(cfg.probe.situations)
        self.instructions = [cfg.probe.instructions] if isinstance(
            cfg.probe.instructions, str) else list(cfg.probe.instructions)

        self.num_samples = max(len(self.sources), len(self.scene_ids),
                               len(self.situations), len(self.instructions))
        if len(self.sources) == 1:
            self.sources = self.sources * self.num_samples
        if len(self.scene_ids) == 1:
            self.scene_ids = self.scene_ids * self.num_samples
        if len(self.situations) == 1:
            self.situations = self.situations * self.num_samples
        if len(self.instructions) == 1:
            self.instructions = self.instructions * self.num_samples

        assert len(self.sources) == len(self.scene_ids) == len(
            self.situations) == len(self.instructions)

        self.data_dict = {
            'source':
            self.sources,
            'scene_id':
            self.scene_ids,
            'prompt_before_obj': [
                self.role_prompt + self.situation_prompt.format(situation=s)
                for s in self.situations
            ],
            'prompt_middle_1': [self.egoview_prompt] * self.num_samples,
            'prompt_middle_2': [self.objects_prompt] * self.num_samples,
            'prompt_after_obj': [],
            'obj_fts': [],
            'obj_masks': [],
            'obj_locs': [],
            'anchor_locs':
            torch.zeros(self.num_samples, 3, device=device),
            'img_fts':
            torch.zeros(self.num_samples, 3, 224, 224, device=device),
            'img_masks':
            torch.zeros(self.num_samples, 1, dtype=torch.bool, device=device),
        }
        for instruction in self.instructions:
            if 'USER:' in instruction:
                # dialogue
                self.data_dict['prompt_after_obj'].append(instruction)
            else:
                # single question
                self.data_dict['prompt_after_obj'].append(
                    self.task_prompt.format(instruction=instruction))

        anchor_orient = torch.zeros(self.num_samples, 4, device=device)
        anchor_orient[:, -1] = 1
        self.data_dict['anchor_orientation'] = anchor_orient

        # load scene
        for source, scene_id in zip(self.sources, self.scene_ids):
            obj_fts, obj_masks, obj_locs = self.load_scene(source, scene_id)
            self.data_dict['obj_fts'].append(obj_fts)
            self.data_dict['obj_masks'].append(obj_masks)
            self.data_dict['obj_locs'].append(obj_locs)

        self.data_dict['obj_fts'] = torch.stack(
            self.data_dict['obj_fts']).to(device)
        self.data_dict['obj_masks'] = torch.stack(
            self.data_dict['obj_masks']).to(device)
        self.data_dict['obj_locs'] = torch.stack(
            self.data_dict['obj_locs']).to(device)

        self.save_dir = os.path.join(self.exp_dir, 'probe')
        iu.make_dir(self.save_dir)
        self.log_path = os.path.join(self.save_dir, 'results.json')
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {}
        if self.pretrained_ckpt_path not in self.log:
            self.log[self.pretrained_ckpt_path] = []

    def load_scene(self, source, scene_id):
        if source.lower() in ['3rscan', 'scannet']:
            if source.lower() == '3rscan':
                obj_pcds = self.load_rscan(scene_id)['obj_pcds']
            elif source.lower() == 'scannet':
                obj_pcds = self.load_scannet(scene_id)['obj_pcds']
            selected_obj_pcds = list(obj_pcds.values())[:self.max_obj_len]
        elif source.lower() == 'objaverse':
            raise NotImplementedError
        elif source.lower() in ['mp3d', 'hm3d']:
            raise NotImplementedError
        elif source.lower() in ['cliport', 'arnold']:
            raise NotImplementedError
        else:
            raise ValueError(f'Unsupported source: {source}')

        obj_fts, obj_locs, _ = self.preprocess_pcd(selected_obj_pcds,
                                                   return_anchor=False)
        obj_fts = pad_tensors(obj_fts, lens=self.max_obj_len,
                              pad=1.0).float()  # O, num_points, 6
        obj_masks = (torch.arange(self.max_obj_len) < len(obj_locs))  # O
        obj_locs = pad_tensors(obj_locs, lens=self.max_obj_len,
                               pad=0.0).float()  # O, 6
        return obj_fts, obj_masks, obj_locs

    @torch.no_grad()
    def run(self):
        for i in trange(ceil(self.num_samples / self.batch_size)):
            batch_data_dict = {}
            for k in self.data_dict.keys():
                batch_data_dict[k] = self.data_dict[k][self.batch_size *
                                                       i:self.batch_size *
                                                       (i + 1)]
            output = self.forward(batch_data_dict, inference=True)
            for j in range(self.batch_size):
                idx = self.batch_size * i + j
                if idx >= self.num_samples:
                    break
                response_log = {
                    'source': self.sources[idx],
                    'scene_id': self.scene_ids[idx],
                    'situation': self.situations[idx],
                    'instruction': self.instructions[idx],
                    'response': output['output_txt'][j],
                }
                logger.info(response_log)
                self.log[self.pretrained_ckpt_path].append(response_log)

                if self.save_obj_tokens:
                    torch.save(
                        {
                            'obj_tokens':
                            output['obj_tokens'][j].unsqueeze(0).cpu(),
                            'obj_masks':
                            output['obj_masks'][j].unsqueeze(0).cpu(),
                        },
                        os.path.join(
                            self.save_dir,
                            f'{self.sources[idx]}-{self.scene_ids[idx]}.pth'))

        with open(self.log_path, 'w') as f:
            json.dump(self.log, f, indent=2)


@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg):
    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        key = str(rgetattr(cfg, name))
        if key:
            naming_keys.append(key)
    exp_name = '_'.join(naming_keys)

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir, exp_name,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        if 'time' in cfg.naming_keywords else '')
    iu.make_dir(cfg.exp_dir)

    prober = LeoProber(cfg)
    prober.run()


if __name__ == '__main__':
    main()
