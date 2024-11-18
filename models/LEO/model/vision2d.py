import os

import numpy as np
import timm
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from einops import rearrange
from model.build import MODULE_REGISTRY
from model.utils import disabled_train

logger = get_logger(__name__)


def simple_conv_and_linear_weights_init(m):
    if type(m) in [
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        simple_linear_weights_init(m)


def simple_linear_weights_init(m):
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


@MODULE_REGISTRY.register()
class GridFeatureExtractor2D(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        init_func_name = '_'.join(
            [cfg.backbone_name, cfg.backbone_pretrain_dataset])
        init_func = globals().get(init_func_name)
        if init_func and callable(init_func):
            self.backbone = init_func(pretrained=cfg.use_pretrain,
                                      freeze=cfg.freeze)
        else:
            raise NotImplementedError(
                f'Backbone2D does not support {init_func_name}')

        self.pooling = cfg.pooling
        if self.pooling:
            if self.pooling == 'avg':
                self.pooling_layers = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                self.out_channels = self.backbone.out_channels
            elif self.pooling == 'conv':
                self.pooling_layers = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels, 64, 1),
                    nn.ReLU(inplace=True), nn.Conv2d(64, 32, 1), nn.Flatten())
                self.pooling_layers.apply(simple_conv_and_linear_weights_init)
                self.out_channels = 32 * 7 * 7  # hardcode for 224x224
            elif self.pooling in ['attn', 'attention']:
                self.visual_attention = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels,
                              self.backbone.out_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.backbone.out_channels,
                              self.backbone.out_channels, 1),
                )
                self.visual_attention.apply(
                    simple_conv_and_linear_weights_init)

                def _attention_pooling(x):
                    B, C, H, W = x.size()
                    attn = self.visual_attention(x)
                    attn = attn.view(B, C, -1)
                    x = x.view(B, C, -1)
                    attn = attn.softmax(dim=-1)
                    x = torch.einsum('b c n, b c n -> b c', x, x)
                    return x

                self.pooling_layers = _attention_pooling
                self.out_channels = self.backbone.out_channels
            else:
                raise NotImplementedError(
                    f'Backbone2D does not support {self.pooling} pooling')
        else:
            self.out_channels = self.backbone.out_channels

        logger.info(f'Build Backbone2D: {init_func_name}, ' +
                    f'pretrain = {cfg.use_pretrain}, freeze = {cfg.freeze}, ' +
                    f'pooling = {self.pooling if self.pooling else None}')

    def forward(self, x):
        if self.pooling:
            x = self.backbone(x, flat_output=False)
            x = self.pooling_layers(x).unsqueeze(1)
            return x
        else:
            return self.backbone(x, flat_output=True)


class Backbone2DWrapper(nn.Module):

    def __init__(self, model, tag, freeze=True):
        super().__init__()
        self.model = model
        self.tag = tag
        self.freeze = freeze
        if 'convnext' in tag:
            self.out_channels = 1024
        elif 'swin' in tag:
            self.out_channels = 1024
        elif 'vit' in tag:
            self.out_channels = 768
        elif 'resnet' in tag:
            self.out_channels = 2048
        else:
            raise NotImplementedError

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            self.train = disabled_train

    def forward_normal(self, x, flat_output=False):
        feat = self.model.forward_features(x)
        if 'swin' in self.tag:
            feat = rearrange(feat, 'b h w c -> b c h w')
        if 'vit_base_32_timm_laion2b' in self.tag or 'vit_base_32_timm_openai' in self.tag:
            # TODO: [CLS] is prepended to the patches.
            feat = rearrange(feat[:, 1:], 'b (h w) c -> b c h w', h=7)
        if flat_output:
            feat = rearrange(feat, 'b c h w -> b (h w) c')
        return feat

    @torch.no_grad()
    def forward_frozen(self, x, flat_output=False):
        return self.forward_normal(x, flat_output)

    def forward(self, x, flat_output=False):
        if self.freeze:
            return self.forward_frozen(x, flat_output)
        else:
            return self.forward_normal(x, flat_output)


# 1024x7x7 or 49x1024
def convnext_base_in1k(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model('convnext_base',
                                               pretrained=pretrained),
                             'convnext_base_in1k',
                             freeze=freeze)


# 1024x7x7 or 49x1024
def convnext_base_in22k(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model('convnext_base_in22k',
                                               pretrained=pretrained),
                             'convnext_base_in22k',
                             freeze=freeze)


# 1024x7x7 or 49x1024
def convnext_base_laion2b(pretrained=False, freeze=True, **kwargs):
    m = timm.create_model('convnext_base.clip_laion2b', pretrained=pretrained)
    if kwargs.get('reset_clip_s2b2'):
        logger.debug(
            'Resetting the last conv layer of convnext-base to random init.')
        s = m.state_dict()
        for i in s.keys():
            if 'stages.3.blocks.2' in i and ('weight' in i or 'bias' in i):
                s[i].normal_()
        m.load_state_dict(s, strict=True)

    return Backbone2DWrapper(m, 'convnext_base_laion2b', freeze=freeze)


# 1024x7x7 or 49x1024
def swin_base_in1k(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model('swin_base_patch4_window7_224',
                                               pretrained=pretrained),
                             'swin_base_timm_in1k',
                             freeze=freeze)


# 1024x7x7 or 49x1024
def swin_base_in22k(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model(
        'swin_base_patch4_window7_224_in22k', pretrained=pretrained),
                             'swin_base_timm_in22k',
                             freeze=freeze)


# 768x7x7 or 49x768
def vit_b_32_laion2b(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model(
        'vit_base_patch32_clip_224.laion2b', pretrained=pretrained),
                             'vit_base_32_timm_laion2b',
                             freeze=freeze)


# 768x7x7 or 49x768
def vit_b_32_openai(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model(
        'vit_base_patch32_clip_224.openai', pretrained=pretrained),
                             'vit_base_32_timm_openai',
                             freeze=freeze)


# 2048x7x7 or 49x2048
def resnet_50_in1k(pretrained=False, freeze=True, **kwargs):
    return Backbone2DWrapper(timm.create_model('resnet50.gluon_in1k',
                                               pretrained=pretrained),
                             'resnet50_timm_in1k',
                             freeze=freeze)
