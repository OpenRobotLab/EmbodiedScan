import math
from typing import Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from model.utils import get_activation_fn
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import (Conv1D,
                                         find_pruneable_heads_and_indices,
                                         prune_conv1d_layer)

logger = get_logger(__name__)


class CrossAttentionLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 k_dim=None,
                 v_dim=None,
                 prenorm=True):
        super().__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.prenorm = prenorm
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    kdim=k_dim,
                                                    vdim=v_dim)
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        if self.prenorm:
            tgt2 = self.norm1(tgt2)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm1(tgt)
        if self.prenorm:
            tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if not self.prenorm:
            tgt = self.norm3(tgt)
        return tgt, cross_attn_matrices


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=True)
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            query=tgt2,
            key=tgt2,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 batch_first=True,
                 dropout=0.1,
                 activation='relu',
                 prenorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=batch_first)
        # Implementation of Feedforward modules
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.prenorm = prenorm

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        if self.prenorm:
            tgt2 = self.norm1(tgt2)
        tgt2, self_attn_matrices = self.self_attn(
            query=tgt2,
            key=tgt2,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        if not self.prenorm:
            tgt = self.norm1(tgt)
        if self.prenorm:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        if not self.prenorm:
            tgt = self.norm2(tgt)
        return tgt, self_attn_matrices


class MultiHeadAttentionSpatial(nn.Module):

    def __init__(
        self,
        d_model,
        n_head,
        dropout=0.1,
        spatial_multihead=True,
        spatial_dim=5,
        spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model,
                                                                   n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(
                d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' %
                                      (self.spatial_attn_fusion))

    def forward(self,
                q,
                k,
                v,
                pairwise_locs,
                key_padding_mask=None,
                txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q),
                             'b l (head k) -> head b l k',
                             head=self.n_head)
        k = einops.rearrange(self.w_ks(k),
                             'b t (head k) -> head b t k',
                             head=self.n_head)
        v = einops.rearrange(self.w_vs(v),
                             'b t (head v) -> head b t v',
                             head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn,
                                         'h b l t -> (h nh) b l t',
                                         nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn,
                                        'b l t (h k) -> h b l t k',
                                        h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(
                q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual)
            spatial_weights = einops.rearrange(spatial_weights,
                                               'b l (h d) -> h b l d',
                                               h=self.spatial_n_head,
                                               d=self.spatial_dim + 1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights,
                                                '1 b l d -> h b l d',
                                                h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights,
                                    pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask,
                                 'b t -> h b l t',
                                 h=self.n_head,
                                 l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) +
                          torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)

        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn


class TransformerSpatialDecoderLayer(TransformerDecoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 spatial_multihead=True,
                 spatial_dim=5,
                 spatial_attn_fusion='mul'):
        super().__init__(d_model,
                         nhead,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation)
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model,
            nhead,
            dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_pairwise_locs: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2,
            tgt2,
            tgt2,
            tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices


class TransformerSpatialEncoderLayer(TransformerEncoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 spatial_multihead=True,
                 spatial_dim=5,
                 spatial_attn_fusion='mul'):
        super().__init__(d_model,
                         nhead,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation)
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model,
            nhead,
            dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self,
        tgt,
        tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        tgt2, self_attn_matrices = self.self_attn(
            tgt2,
            tgt2,
            tgt2,
            tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, self_attn_matrices


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class LlamaRotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len /
                                 self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(self.dim /
                                                             (self.dim - 2))
            inv_freq = 1.0 / (base**(
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Attention
# Add option for RoPE
class GPT2Attention(nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.bool)).view(1, 1, max_positions,
                                                   max_positions),
            persistent=False,
        )
        self.register_buffer('masked_bias',
                             torch.tensor(-1e4),
                             persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.use_rope = config.use_rope
        if self.use_rope:
            self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor)
            elif scaling_type == 'dynamic':
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size //
                           self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1)**0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value,
                                    dtype=attn_weights.dtype).to(
                                        attn_weights.device)
            attn_weights = torch.where(causal_mask,
                                       attn_weights.to(attn_weights.dtype),
                                       mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self,
                                   query,
                                   key,
                                   value,
                                   attention_mask=None,
                                   head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads,
                                   q_seq_len,
                                   k_seq_len,
                                   dtype=torch.float32,
                                   device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1))**0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off torch.cuda.amp.autocast) and reorder (Scale K by 1 / root(dk))
        with torch.cuda.amp.autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len,
                                 dk), key.transpose(-1, -2).reshape(
                                     -1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights,
                                         q.float(),
                                         k.float(),
                                         beta=0,
                                         alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len,
                                                k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
                attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                'Error with upcasting, attn_weights does not have dtype torch.float32'
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Splits hidden_size dim into attn_head_size and num_heads."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1,
                              3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merges attn_head_size dim and num_attn_heads dim into
        hidden_size."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size, )
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, 'q_attn'):
                raise ValueError(
                    'If class is used as cross attention, the weights `q_attn` have to be defined. '
                    'Please make sure to instantiate class with `CausalAttention(..., is_cross_attention=True)`.'
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(
                self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # only add RoPE in self-attention
        # add RoPE embedding before concating past kv (they already have RoPE)
        if encoder_hidden_states is None and self.use_rope:
            kv_seq_len = key.shape[-2]
            if layer_past is not None:
                kv_seq_len += layer_past[0].shape[-2]
            cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
            query, key = apply_rotary_pos_emb(query, key, cos, sin,
                                              position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value,
                                                   attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights, )

        return outputs  # a, present, (attentions)


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP
class GPT2MLP(nn.Module):

    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Block
# Add option for RoPE
class GPT2Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config,
                                                is_cross_attention=True,
                                                layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size,
                                              eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[
            torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, 'crossattention'):
                raise ValueError(
                    f'If `encoder_hidden_states` are passed, {self} has to be instantiated with '
                    'cross-attention layers by setting `config.add_cross_attention=True`'
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[
                2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if not config.use_rope:
            self.wpe = nn.Embedding(config.max_position_embeddings,
                                    self.embed_dim)
        else:
            self.wpe = None

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPT2Block(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if 'c_proj' in name and 'weight' in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0,
                               std=(self.config.initializer_range /
                                    math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPT2Model):
            module.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2Model.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length,
                                        input_shape[-1] + past_length,
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape,
                                                    device=device)
            encoder_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if self.wpe:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1), )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
        ) if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device)
                        for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1], )

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1], )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2], )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and 'cuda:' + str(k) != self.last_device:
                        hidden_states = hidden_states.to('cuda:' + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states, presents, all_hidden_states,
                all_self_attentions, all_cross_attentions
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
