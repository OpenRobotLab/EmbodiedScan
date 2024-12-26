import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from model.build import MODULE_REGISTRY
from model.pcd_backbone import PointcloudBackbone
from model.transformers import (TransformerEncoderLayer,
                                TransformerSpatialEncoderLayer)
from model.utils import _init_weights_bert, calc_pairwise_locs, layer_repeat

logger = get_logger(__name__)


def generate_fourier_features(pos,
                              num_bands=10,
                              max_freq=15,
                              concat_pos=True,
                              sine_only=False):
    # Input: B, N, C
    # Output: B, N, C'
    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq,
                                end=max_freq,
                                steps=num_bands,
                                device=device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1,
                                                num_bands) * freq_bands
    per_pos_features = torch.reshape(
        per_pos_features,
        [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat([
            torch.sin(np.pi * per_pos_features),
            torch.cos(np.pi * per_pos_features)
        ],
                                     dim=-1)
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


@MODULE_REGISTRY.register()
class OSE3D(nn.Module):
    # Open-vocabulary, Spatial-attention, Embodied-token, 3D-agent
    def __init__(self, cfg):
        super().__init__()
        self.use_spatial_attn = cfg.use_spatial_attn  # spatial attention
        self.use_embodied_token = cfg.use_embodied_token  # embodied token
        hidden_dim = cfg.hidden_dim

        # pcd backbone
        self.obj_encoder = PointcloudBackbone(cfg.backbone)
        self.obj_proj = nn.Linear(self.obj_encoder.out_dim, hidden_dim)

        # embodied token
        if self.use_embodied_token:
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3))
        self.orient_encoder = nn.Linear(cfg.fourier_size, hidden_dim)
        self.obj_type_embed = nn.Embedding(2, hidden_dim)

        # spatial encoder
        if self.use_spatial_attn:
            spatial_encoder_layer = TransformerSpatialEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
                spatial_dim=cfg.spatial_encoder.spatial_dim,
                spatial_multihead=cfg.spatial_encoder.spatial_multihead,
                spatial_attn_fusion=cfg.spatial_encoder.spatial_attn_fusion,
            )
        else:
            spatial_encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=cfg.spatial_encoder.dim_feedforward,
                dropout=cfg.spatial_encoder.dropout,
                activation=cfg.spatial_encoder.activation,
            )

        self.spatial_encoder = layer_repeat(
            spatial_encoder_layer,
            cfg.spatial_encoder.num_layers,
        )
        self.pairwise_rel_type = cfg.spatial_encoder.pairwise_rel_type
        self.spatial_dist_norm = cfg.spatial_encoder.spatial_dist_norm
        self.spatial_dim = cfg.spatial_encoder.spatial_dim
        self.obj_loc_encoding = cfg.spatial_encoder.obj_loc_encoding

        # location encoding
        if self.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.obj_loc_encoding == 'diff_all':
            num_loc_layers = cfg.spatial_encoder.num_layers

        loc_layer = nn.Sequential(
            nn.Linear(cfg.spatial_encoder.dim_loc, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)

        #        logger.info("Build 3D module: OSE3D")

        # only initialize spatial encoder and loc layers
        self.spatial_encoder.apply(_init_weights_bert)
        self.loc_layers.apply(_init_weights_bert)

        if self.use_embodied_token:
            nn.init.normal_(self.anchor_feat, std=0.02)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, data_dict):
        """data_dict requires keys:

        obj_fts: (B, N, P, 6), xyz + rgb
        obj_masks: (B, N), 1 valid and 0 masked
        obj_locs: (B, N, 6), xyz + whd
        anchor_locs: (B, 3)
        anchor_orientation: (B, C)
        """

        obj_feats = self.obj_encoder(data_dict['obj_fts'])
        obj_feats = self.obj_proj(obj_feats)
        obj_masks = ~data_dict[
            'obj_masks']  # flipped due to different convention of TransformerEncoder

        B, N = obj_feats.shape[:2]
        device = obj_feats.device

        obj_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
        obj_type_embeds = self.obj_type_embed(obj_type_ids)

        if self.use_embodied_token:
            # anchor feature
            anchor_orient = data_dict['anchor_orientation'].unsqueeze(1)
            anchor_orient_feat = self.orient_encoder(
                generate_fourier_features(anchor_orient))
            anchor_feat = self.anchor_feat + anchor_orient_feat
            anchor_mask = torch.zeros((B, 1), dtype=bool, device=device)

            # anchor loc (3) + size (3)
            anchor_loc = torch.cat([
                data_dict['anchor_locs'].unsqueeze(1),
                self.anchor_size.expand(B, -1, -1).to(device)
            ],
                                   dim=-1)

            # anchor type
            anchor_type_id = torch.ones((B, 1),
                                        dtype=torch.long,
                                        device=device)
            anchor_type_embed = self.obj_type_embed(anchor_type_id)

            # fuse anchor and objs
            all_obj_feats = torch.cat([anchor_feat, obj_feats], dim=1)
            all_obj_masks = torch.cat((anchor_mask, obj_masks), dim=1)

            all_obj_locs = torch.cat([anchor_loc, data_dict['obj_locs']],
                                     dim=1)
            all_obj_type_embeds = torch.cat(
                (anchor_type_embed, obj_type_embeds), dim=1)

        else:
            all_obj_feats = obj_feats
            all_obj_masks = obj_masks

            all_obj_locs = data_dict['obj_locs']
            all_obj_type_embeds = obj_type_embeds

        all_obj_feats = all_obj_feats + all_obj_type_embeds

        # call spatial encoder
        if self.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                all_obj_locs[:, :, :3],
                all_obj_locs[:, :, 3:],
                pairwise_rel_type=self.pairwise_rel_type,
                spatial_dist_norm=self.spatial_dist_norm,
                spatial_dim=self.spatial_dim,
            )

        for i, pc_layer in enumerate(self.spatial_encoder):
            if self.obj_loc_encoding == 'diff_all':
                query_pos = self.loc_layers[i](all_obj_locs)
            else:
                query_pos = self.loc_layers[0](all_obj_locs)
            if not (self.obj_loc_encoding == 'same_0' and i > 0):
                all_obj_feats = all_obj_feats + query_pos

            if self.use_spatial_attn:
                all_obj_feats, _ = pc_layer(all_obj_feats,
                                            pairwise_locs,
                                            tgt_key_padding_mask=all_obj_masks)
            else:
                all_obj_feats, _ = pc_layer(all_obj_feats,
                                            tgt_key_padding_mask=all_obj_masks)

        data_dict['obj_tokens'] = all_obj_feats
        data_dict['obj_masks'] = ~all_obj_masks

        return data_dict
