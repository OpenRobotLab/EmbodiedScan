import contextlib
import copy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def maybe_autocast(model, dtype='bf16', enabled=True):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = model.device != torch.device('cpu')

    if dtype == 'bf16':
        dtype = torch.bfloat16
    elif dtype == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)
    else:
        return contextlib.nullcontext()


def _init_weights_bert(module, std=0.02):
    """Huggingface transformer weight initialization, most commonly for bert
    initialization."""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


#########################################################
# General modules helpers
#########################################################
def get_activation_fn(activation_type):
    if activation_type not in ['relu', 'gelu', 'glu']:
        raise RuntimeError(
            f'activation function currently support relu/gelu, not {activation_type}'
        )
    return getattr(F, activation_type)


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(*[
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.LayerNorm(hidden_size, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    ])


def layer_repeat(module, N):
    return nn.ModuleList([copy.deepcopy(module)
                          for _ in range(N - 1)] + [module])


#########################################################
# Specific modules helpers
#########################################################
def calc_pairwise_locs(obj_centers,
                       obj_whls,
                       eps=1e-10,
                       pairwise_rel_type='center',
                       spatial_dist_norm=True,
                       spatial_dim=5):
    if pairwise_rel_type == 'mlp':
        obj_locs = torch.cat([obj_centers, obj_whls], 2)
        pairwise_locs = torch.cat([
            einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
            einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))
        ],
                                  dim=3)
        return pairwise_locs

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) +
                                eps)  # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1),
                              dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(
            max_dists, 'b -> b 1 1')
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)

    pairwise_dists_2d = torch.sqrt(
        torch.sum(pairwise_locs[..., :2]**2, 3) + eps)
    if pairwise_rel_type == 'center':
        pairwise_locs = torch.stack([
            norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
            pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] /
            pairwise_dists_2d, pairwise_locs[..., 0] / pairwise_dists_2d
        ],
                                    dim=3)
    elif pairwise_rel_type == 'vertical_bottom':
        bottom_centers = torch.clone(obj_centers)
        bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
        bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                               - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
        bottom_pairwise_dists = torch.sqrt(
            torch.sum(bottom_pairwise_locs**2, 3) + eps)  # (b, l, l)
        bottom_pairwise_dists_2d = torch.sqrt(
            torch.sum(bottom_pairwise_locs[..., :2]**2, 3) + eps)
        pairwise_locs = torch.stack([
            norm_pairwise_dists, bottom_pairwise_locs[..., 2] /
            bottom_pairwise_dists, bottom_pairwise_dists_2d /
            bottom_pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
            pairwise_locs[..., 0] / pairwise_dists_2d
        ],
                                    dim=3)

    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs
