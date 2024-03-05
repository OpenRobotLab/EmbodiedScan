# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from embodiedscan.utils import ConfigType, OptConfigType

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class PositionEmbeddingLearned(BaseModule):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, embed_dims=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, embed_dims, kernel_size=1),
            nn.BatchNorm1d(embed_dims), nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2).contiguous()


class SparseFeatureFusionTransformerDecoderLayer(BaseModule):

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(embed_dims=256,
                                                     num_heads=8,
                                                     dropout=0.0,
                                                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(embed_dims=256,
                                                      num_heads=8,
                                                      dropout=0.0,
                                                      batch_first=True),
                 cross_attn_text_cfg: OptConfigType = dict(embed_dims=256,
                                                           num_heads=8,
                                                           dropout=0.0,
                                                           batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_text_cfg = cross_attn_text_cfg
        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True

        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)
        self.self_posembed = PositionEmbeddingLearned(3, self.embed_dims)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        # self attention  dropout is down in the self_attn layer
        query = self.self_attn(query=query,
                               key=query,
                               value=query,
                               query_pos=query_pos,
                               key_pos=query_pos,
                               attn_mask=self_attn_mask,
                               **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(query=query,
                                     query_pos=query_pos,
                                     key=memory_text,
                                     value=memory_text,
                                     key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and point cloud
        query = self.cross_attn(query=query,
                                key=key,
                                value=value,
                                query_pos=query_pos,
                                key_pos=key_pos,
                                attn_mask=cross_attn_mask,
                                key_padding_mask=key_padding_mask,
                                **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class SparseFeatureFusionTransformerDecoder(BaseModule):
    """Decoder of DETR.

    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`,
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            SparseFeatureFusionTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.self_posembed = PositionEmbeddingLearned(9, self.embed_dims)
        self.cross_posembed = PositionEmbeddingLearned(3, self.embed_dims)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Tensor, self_attn_mask: Tensor,
                cross_attn_mask: Tensor, query_coords: Tensor,
                key_coords: Tensor, pred_bboxes: Tensor, text_feats: Tensor,
                text_attention_mask: Tensor, bbox_head: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            pred_sizes (Tensor): The initial reference, has shape
                (bs, num_queries, 3 or 6) with the last dimension arranged as
                (x, y, z) or (dx, dy, dz).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - pred_sizes (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_bboxes = []
        for lid, layer in enumerate(self.layers):

            query_pos = self.self_posembed(pred_bboxes)
            key_pos = self.cross_posembed(key_coords)
            query = layer(query=query,
                          key=key,
                          value=value,
                          query_pos=query_pos,
                          key_pos=key_pos,
                          memory_text=text_feats,
                          self_attn_mask=self_attn_mask,
                          cross_attn_mask=cross_attn_mask,
                          key_padding_mask=key_padding_mask,
                          text_attention_mask=text_attention_mask,
                          **kwargs)

            if bbox_head is not None:
                # (bs, num_query, 9)
                bbox_preds = bbox_head.reg_branches[lid](query)
                new_pred_bboxes = bbox_head._bbox_pred_to_bbox(
                    query_coords, bbox_preds)
                pred_bboxes = new_pred_bboxes.detach().clone()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_bboxes.append(new_pred_bboxes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_bboxes)

        return query, new_pred_bboxes
