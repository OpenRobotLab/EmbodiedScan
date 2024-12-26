"""Adapted from PointNeXt implementation in repo:

https://github.com/guochengqian/openpoints.
"""
from typing import List

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from einops import rearrange
from model.pointnext.layers import (CHANNEL_MAP, create_act,
                                    create_convblock1d, create_convblock2d,
                                    create_grouper, furthest_point_sample,
                                    get_aggregation_feautres, random_sample,
                                    three_interpolation)

logger = get_logger(__name__)


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set Set abstraction layer abstracts
    features from a larger set to a smaller set Local aggregation layer
    aggregates features from the same set."""

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={
                     'NAME': 'ballquery',
                     'radius': 0.1,
                     'nsample': 16
                 },
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs):
        super().__init__()
        if kwargs:
            logger.warning(
                f'kwargs: {kwargs} are not used in {__class__.__name__}')
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # number of layers in each block
            convs.append(
                create_convblock2d(channels[i],
                                   channels[i + 1],
                                   norm_args=norm_args,
                                   act_args=None if i == (len(channels) - 2)
                                   and not last_act else act_args,
                                   **conv_args))
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # neighborhood_features
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)
        f = self.pool(self.convs(fj))
        """ DEBUG neighbor numbers.
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logger.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual
    connection support."""

    def __init__(
        self,
        in_channels,
        out_channels,
        layers=1,
        stride=1,
        group_args={
            'NAME': 'ballquery',
            'radius': 0.1,
            'nsample': 16
        },
        norm_args={'norm': 'bn1d'},
        act_args={'act': 'relu'},
        conv_args=None,
        sampler='fps',
        feature_type='dp_fj',
        use_res=False,
        is_head=False,
        **kwargs,
    ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](
            channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None
            ) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(
                create_conv(channels[i],
                            channels[i + 1],
                            norm_args=norm_args if not is_head else None,
                            act_args=None if i == len(channels) - 2 and
                            (self.use_res or is_head) else act_args,
                            **conv_args))
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers.
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logger.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(f, -1,
                                  idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p,
                                          dp,
                                          fi,
                                          fj,
                                          feature_type=self.feature_type)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++"""

    def __init__(self,
                 mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(nn.Linear(mlp[0], mlp[1]),
                                         nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(
                    create_convblock1d(mlp[i],
                                       mlp[i + 1],
                                       norm_args=norm_args,
                                       act_args=act_args))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(
                    create_convblock1d(mlp[i],
                                       mlp[i + 1],
                                       norm_args=norm_args,
                                       act_args=act_args))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat((f, self.linear2(f_global).unsqueeze(-1).expand(
                -1, -1, f.shape[-1])),
                          dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):

    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={
                     'feature_type': 'dp_fj',
                     'reduction': 'max'
                 },
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation(
            [in_channels, in_channels],
            norm_args=norm_args,
            act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args,
            conv_args=conv_args,
            **aggr_args,
            **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(
                create_convblock1d(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=act_args if
                    (i != len(channels) - 2) and not less_act else None,
                    **conv_args))
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={
                     'feature_type': 'dp_fj',
                     'reduction': 'max'
                 },
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation(
            [in_channels, in_channels, mid_channels, in_channels],
            norm_args=norm_args,
            act_args=None,
            group_args=group_args,
            conv_args=conv_args,
            **aggr_args,
            **kwargs)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class PointNext(nn.Module):

    def __init__(
        self,
        in_channels=3,
        width=32,
        num_blocks=[1, 4, 7, 4, 4],
        strides=[1, 4, 4, 4, 4],
        block='InvResMLP',
        nsample=32,
        radius=0.1,
        conv_args=None,
        aggr_args={
            'feature_type': 'dp_fj',
            'reduction': 'max'
        },
        group_args={
            'NAME': 'ballquery',
            'radius': 0.1,
            'nsample': 32
        },
        norm_args={'norm': 'bn'},
        act_args={'act': 'relu'},
        sampler='fps',
        expansion=4,
        sa_layers=1,
        sa_use_res=False,
        use_res=True,
        radius_scaling=2,
        nsample_scaling=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        block = eval(block)
        self.num_blocks = num_blocks
        self.strides = strides

        self.aggr_args = aggr_args
        self.norm_args = norm_args
        self.act_args = act_args
        self.conv_args = conv_args
        self.sampler = sampler
        self.expansion = expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = use_res
        self.radius = radius

        self.radii = self._to_full_list(self.radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logger.debug(
            f'PointNext: radius = {self.radii}, nsample = {self.nsample}')

        # double width after downsampling
        channels = []
        for stride in self.strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(self.num_blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(
                self._make_enc(block,
                               channels[i],
                               self.num_blocks[i],
                               stride=self.strides[i],
                               group_args=group_args,
                               is_head=i == 0 and self.strides[i] == 1))
        self.encoder = nn.Sequential(*encoder)
        self.out_dim = channels[-1]
        self.channel_list = channels
        self.pool = get_reduction_fn(self.aggr_args.reduction)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _to_full_list(self, param, param_scaling=1):
        # param can be : radius, nsample
        param_list = []
        if isinstance(param, list):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, list) else value
                if len(value) != self.num_blocks[i]:
                    value += [value[-1]] * (self.num_blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial radius is provided), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.num_blocks[i])
                else:
                    param_list.append([param] + [param * param_scaling] *
                                      (self.num_blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self,
                  block,
                  channels,
                  num_blocks,
                  stride,
                  group_args,
                  is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(
            SetAbstraction(self.in_channels,
                           channels,
                           self.sa_layers if not is_head else 1,
                           stride,
                           group_args=group_args,
                           sampler=self.sampler,
                           norm_args=self.norm_args,
                           act_args=self.act_args,
                           conv_args=self.conv_args,
                           is_head=is_head,
                           use_res=self.sa_use_res,
                           **self.aggr_args))
        self.in_channels = channels
        for i in range(1, num_blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(
                block(self.in_channels,
                      aggr_args=self.aggr_args,
                      norm_args=self.norm_args,
                      act_args=self.act_args,
                      group_args=group_args,
                      conv_args=self.conv_args,
                      expansion=self.expansion,
                      use_res=self.use_res))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, x):
        p0 = x[..., :3].to(self.device)
        f0 = x[..., 3:].to(self.device)

        if x.ndim == 4:
            # (batch_size, num_objects, num_points, num_channels)
            batch_size = x.shape[0]
            p0 = rearrange(p0, 'b o p d -> (b o) p d').contiguous()
            f0 = rearrange(f0, 'b o p c -> (b o) c p').contiguous()

            for i in range(len(self.encoder)):
                p0, f0 = self.encoder[i]([p0, f0])
            f0 = self.pool(f0)
            return rearrange(f0, '(b o) c -> b o c', b=batch_size)

        elif x.ndim == 3:
            # (batch_size * num_objects, num_points, num_channels)
            p0 = p0.contiguous()
            f0 = rearrange(f0, 'bo p c -> bo c p').contiguous()

            for i in range(len(self.encoder)):
                p0, f0 = self.encoder[i]([p0, f0])
            f0 = self.pool(f0)
            return f0

        else:
            raise ValueError(
                'Point cloud input shape incorrect, ndim should be either 3 or 4.'
            )

    def forward_seg_feat(self, x):
        p0 = x[..., :3].to(self.device)
        f0 = x[..., 3:].to(self.device)

        if x.ndim == 4:
            # (batch_size, num_objects, num_points, num_channels)
            batch_size = x.shape[0]
            p0 = rearrange(p0, 'b o p d -> (b o) p d').contiguous()
            f0 = rearrange(f0, 'b o p c -> (b o) c p').contiguous()

            p, f = [p0], [f0]
            for i in range(len(self.encoder)):
                _p, _f = self.encoder[i]([p[-1], f[-1]])
                p.append(_p)
                f.append(_f)
            for i in range(len(p)):
                p[i] = rearrange(p[i], '(b o) p d -> b o p d', b=batch_size)
                f[i] = rearrange(f[i], '(b o) c p -> b o c p', b=batch_size)
            return p, f

        elif x.ndim == 3:
            # (batch_size * num_objects, num_points, num_channels)
            p0 = p0.contiguous()
            f0 = rearrange(f0, 'bo p c -> bo c p').contiguous()

            p, f = [p0], [f0]
            for i in range(len(self.encoder)):
                _p, _f = self.encoder[i]([p[-1], f[-1]])
                p.append(_p)
                f.append(_f)
            return p, f

        else:
            raise ValueError(
                'Point cloud input shape incorrect, ndim should be either 3 or 4.'
            )

    def forward(self, x):
        return self.forward_cls_feat(x)
