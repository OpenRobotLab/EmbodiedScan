import os

import torch
from accelerate.logging import get_logger
from einops import rearrange
from hydra.utils import instantiate
# from model.pointnext.pointnext import PointNext
from model.pointbert.pointbert import PointBERT
from model.pointnetpp.pointnetpp import PointNetPP
from model.utils import disabled_train
from torch import nn

logger = get_logger(__name__)


class PointcloudBackbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.pcd_net = instantiate(cfg.net)
        self.backbone_name = cfg.net._target_.split('.')[-1]
        self.out_dim = self.pcd_net.out_dim
        #        logger.info(f"Build PointcloudBackbone: {self.backbone_name}")

        path = cfg.path
        if path is not None and os.path.exists(path):
            self.pcd_net.load_state_dict(torch.load(path), strict=False)
            logger.info(f'Load {self.backbone_name} weights from {path}')

        self.freeze = cfg.freeze
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            self.train = disabled_train


#            logger.info(f"Freeze {self.backbone_name}")

    def forward_normal(self, obj_pcds):
        # obj_pcds: (batch_size, num_objs, num_points, 6)
        batch_size = obj_pcds.shape[0]
        obj_embeds = self.pcd_net(rearrange(obj_pcds, 'b o p d -> (b o) p d'))
        obj_embeds = rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        return obj_embeds

    @torch.no_grad()
    def forward_frozen(self, obj_pcds):
        return self.forward_normal(obj_pcds)

    def forward(self, obj_pcds):
        if self.freeze:
            return self.forward_frozen(obj_pcds)
        else:
            return self.forward_normal(obj_pcds)
