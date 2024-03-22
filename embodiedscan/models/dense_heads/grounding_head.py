# Copyright (c) OpenRobotLab. All rights reserved.
import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig, reduce_mean
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from pytorch3d.transforms import matrix_to_euler_angles
from torch import Tensor

from embodiedscan.registry import MODELS, TASK_UTILS
from embodiedscan.structures import (EulerDepthInstance3DBoxes,
                                     rotation_3d_in_axis, rotation_3d_in_euler)
from embodiedscan.utils.typing_config import SampleList


class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.

          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 max_text_len: int = 256,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(torch.Tensor([float(log_scale)]),
                                          requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(torch.Tensor([bias_value]),
                                     requires_grad=True)

    def forward(self,
                visual_feat: Tensor,
                text_feat: Tensor,
                text_token_mask: Tensor,
                visual_feat_mask: Tensor = None) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.  # (b, num_query, dim)
            text_feat (Tensor): Text features.  # (b, text_lenth, text_dim)
            text_token_mask (Tensor): A mask used for text feats.
            visual_feat_mask (Tensor, optional): Mask used for visual features.
                Defaults to None.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(
            -1, -2)  # (b, num_query, text_lenth)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        # fill -inf in the padding part
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        if visual_feat_mask is not None:
            res.masked_fill_(~visual_feat_mask[:, :, None], float('-inf'))

        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res


@MODELS.register_module()
class GroundingHead(BaseModule):
    """3D Grounding Head."""

    def __init__(self,
                 num_classes: int,
                 embed_dims: int = 256,
                 num_pred_layer: int = 7,
                 num_reg_fcs: int = 2,
                 num_reg: int = 9,
                 box_coder: str = 'baseline',
                 sync_cls_avg_factor: bool = False,
                 decouple_bbox_loss: bool = False,
                 decouple_groups: int = 3,
                 decouple_weights: Optional[list] = None,
                 norm_decouple_loss: bool = False,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss',
                                             bg_cls_weight=0.1,
                                             use_sigmoid=False,
                                             loss_weight=1.0,
                                             class_weight=1.0),
                 loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
                 train_cfg: ConfigType = dict(assigner=dict(
                     type='HungarianAssigner3D',
                     match_costs=[
                         dict(type='ClassificationCost', weight=1.),
                         dict(type='BBoxL1Cost', weight=5.0,
                              box_format='xywh'),
                         dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     ])),
                 contrastive_cfg=dict(max_text_len=256),
                 share_pred_layer: bool = False,
                 test_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        self.contrastive_cfg = contrastive_cfg
        self.max_text_len = contrastive_cfg.get('max_text_len', 256)
        super().__init__(init_cfg=init_cfg)
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.decouple_bbox_loss = decouple_bbox_loss
        self.decouple_groups = decouple_groups
        self.norm_decouple_loss = norm_decouple_loss
        if decouple_weights is None:
            self.decouple_weights = [
                1.0 / self.decouple_groups for _ in range(self.decouple_groups)
            ]
        else:
            self.decouple_weights = decouple_weights
        self.num_reg = num_reg
        self.box_coder = box_coder
        assert self.box_coder in ('baseline', 'FCAF')
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is GroundingHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.num_reg))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

    def get_targets(self, cls_scores_list: List[Tensor],
                    pred_bboxes_list: List[Tensor],
                    batch_gt_instances_3d: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                                     cls_scores_list,
                                                     pred_bboxes_list,
                                                     batch_gt_instances_3d)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _bbox_pred_to_bbox(self, points, bbox_pred: Tensor) -> Tensor:
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, num_query, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape
                (N, num_query, 12) or (N, 9) or (N, 12), i.e.,
                for baseline box_coder:
                9-dim: (3D offsets to the center, log(3D sizes),
                alpha, beta, gamma)
                12-dim: (3D offsets to the center, log(3D sizes),
                x_raw (3D vector), y_raw (3D vector));
                for FCAF box_coder:
                9-dim: (log(distances to 6 faces) (6D vector),
                alpha, beta, gamma),
                12-dim: (log(distances to 6 faces) (6D vector),
                x_raw (3D vector), y_raw (3D vector)).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7) or (N, 9).
        """

        assert len(points.size()) == len(bbox_pred.size()) == 3
        batch_size = points.shape[0]
        num_queries = points.shape[1]

        if self.box_coder == 'baseline':
            if bbox_pred.shape[-1] == 9:
                center = bbox_pred[..., :3] + points
                size = torch.exp(bbox_pred[..., 3:6]).clamp(min=2e-2)
                euler = bbox_pred[..., 6:]
            elif bbox_pred.shape[-1] == 12:
                center = bbox_pred[..., :3] + points
                size = torch.exp(bbox_pred[..., 3:6]).clamp(min=2e-2)
                x_raw, y_raw = bbox_pred[..., 6:9], bbox_pred[..., 9:]
                rot_mat = ortho_6d_2_Mat(x_raw.view(-1, 3), y_raw.view(-1, 3))
                euler = matrix_to_euler_angles(rot_mat, 'ZXY').view(
                    batch_size, num_queries, 3)
            else:
                raise NotImplementedError
            return torch.cat((center, size, euler), dim=-1)
        elif self.box_coder == 'FCAF':
            if bbox_pred.shape[0] == 0:
                return bbox_pred
            if len(points.size()) == 3:
                points = points.reshape(-1, points.size(-1))
                bbox_pred = bbox_pred.reshape(-1, bbox_pred.size(-1))
            # axis-aligned case
            if bbox_pred.shape[1] == 6:
                x_center = points[..., 0] + (bbox_pred[..., 1] -
                                             bbox_pred[..., 0]) / 2
                y_center = points[..., 1] + (bbox_pred[..., 3] -
                                             bbox_pred[..., 2]) / 2
                z_center = points[..., 2] + (bbox_pred[..., 5] -
                                             bbox_pred[..., 4]) / 2
                # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max
                # -> x, y, z, w, l, h
                base_bbox = torch.stack([
                    x_center,
                    y_center,
                    z_center,
                    bbox_pred[..., 0] + bbox_pred[..., 1],
                    bbox_pred[..., 2] + bbox_pred[..., 3],
                    bbox_pred[..., 4] + bbox_pred[..., 5],
                ], -1)
                return base_bbox
            # for rotated boxes (7-DoF or 9-DoF)
            # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, alpha ->
            # x_center, y_center, z_center, w, l, h, alpha
            # (N, num_queries, 3)
            bbox_pred[..., :6] = torch.exp(bbox_pred[..., :6]).clamp(min=2e-2)
            shift = torch.stack(((bbox_pred[..., 1] - bbox_pred[..., 0]) / 2,
                                 (bbox_pred[..., 3] - bbox_pred[..., 2]) / 2,
                                 (bbox_pred[..., 5] - bbox_pred[..., 4]) / 2),
                                dim=-1).view(-1, 1, 3)
            if bbox_pred.shape[-1] == 7:
                euler = bbox_pred[..., [6]]
                shift = rotation_3d_in_axis(shift, bbox_pred[..., 6],
                                            axis=2)[:, 0, :]
            elif bbox_pred.shape[-1] == 9:
                euler = bbox_pred[..., 6:]
                shift = rotation_3d_in_euler(shift, bbox_pred[..., 6:])[:,
                                                                        0, :]
            elif bbox_pred.shape[-1] == 12:
                x_raw, y_raw = bbox_pred[..., 6:9], bbox_pred[..., 9:]
                rot_mat = ortho_6d_2_Mat(x_raw.view(-1, 3), y_raw.view(-1, 3))
                euler = matrix_to_euler_angles(rot_mat, 'ZXY')
                shift = rotation_3d_in_euler(shift, euler)[:, 0, :]
            center = points + shift
            size = torch.stack(
                (bbox_pred[..., 0] + bbox_pred[..., 1], bbox_pred[..., 2] +
                 bbox_pred[..., 3], bbox_pred[..., 4] + bbox_pred[..., 5]),
                dim=-1)
            return torch.cat((center, size, euler),
                             dim=-1).view(batch_size, num_queries, -1)
        else:
            raise NotImplementedError

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances_3d: InstanceData) -> tuple:
        """Compute regression and classification targets for one sample.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances_3d (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)

        bbox_3d = EulerDepthInstance3DBoxes(bbox_pred)
        pred_instances_3d = InstanceData(scores_3d=cls_score,
                                         bboxes_3d=bbox_3d)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances_3d=pred_instances_3d,
            gt_instances_3d=gt_instances_3d)
        gt_bboxes = gt_instances_3d.bboxes_3d.tensor

        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_bboxes.new_full((num_bboxes, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        # (num_bboxes , max_text_len) gt_labels map
        labels[pos_inds] = gt_instances_3d.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward(
        self,
        hidden_states: Tensor,
        text_feats: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            pred_bboxes (List[Tensor]): List of the reference from the decoder.
                 each with shape (bs, num_queries, 9)
            text_feats (Tensor): Text feats. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_cls_scores = []

        for layer_id in range(hidden_states.shape[0]):
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            cls_scores = self.cls_branches[layer_id](hidden_state, text_feats,
                                                     text_token_mask)
            all_layers_cls_scores.append(cls_scores)

        all_layers_cls_scores = torch.stack(all_layers_cls_scores)

        return (all_layers_cls_scores, )

    def predict(self, hidden_states: Tensor, all_layers_pred_bboxes: Tensor,
                text_feats: Tensor, text_token_mask: Tensor,
                batch_data_samples: SampleList) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            text_feats (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_gt_bboxes_3d = [
            data_samples.gt_instances_3d.bboxes_3d
            for data_samples in batch_data_samples
        ]
        batch_positive_maps = [
            data_samples.gt_instances_3d.positive_maps
            for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = None

        outs = self(hidden_states, text_feats, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            all_layers_pred_bboxes,
            batch_input_metas=batch_input_metas,
            batch_gt_bboxes_3d=batch_gt_bboxes_3d,
            batch_positive_maps=batch_positive_maps,
            batch_token_positive_maps=batch_token_positive_maps)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_pred_bboxes: Tensor,
                        batch_input_metas: List[Dict],
                        batch_gt_bboxes_3d: List,
                        batch_positive_maps: List,
                        batch_token_positive_maps=None) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                max_text_lenth).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 12-tensor with shape (num_decoder_layers, bs,
                num_queries, reg_num).
            batch_input_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.  Actually batch_data_samples

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_pred_bboxes[-1]
        result_list = []
        for img_id in range(len(batch_input_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            gt_bboxes_3d = batch_gt_bboxes_3d[img_id]
            positive_maps = batch_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   gt_bboxes_3d, positive_maps)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                gt_bboxes_3d: Tensor,
                                positive_maps: Tensor) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries

        cls_score = cls_score.sigmoid()  # (num_query, self.max_text_len 256)
        target_token_maps = positive_maps.squeeze(0) > 0
        # (num_query, num_target_tokens)
        target_cls_score = cls_score[:, target_token_maps]
        scores, _ = cls_score.max(-1)
        target_scores = target_cls_score.sum(-1)

        results = InstanceData()
        results.bboxes_3d = EulerDepthInstance3DBoxes(bbox_pred)
        results.scores_3d = scores
        results.target_scores_3d = target_scores

        return results

    def loss(self, hidden_states: Tensor, all_layers_pred_bboxes: Tensor,
             text_feats: Tensor, text_token_mask: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            text_feats (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances_3d = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)

        outs = self(hidden_states, text_feats, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (all_layers_pred_bboxes, batch_gt_instances_3d,
                              batch_input_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_pred_bboxes: Tensor,
        batch_gt_instances_3d: InstanceList,
        batch_input_metas: List[dict],
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_pred_bboxes,
            batch_gt_instances_3d=batch_gt_instances_3d,
            batch_input_metas=batch_input_metas)
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in \
                zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, pred_bboxes: Tensor,
                            batch_gt_instances_3d: InstanceList,
                            batch_input_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all sample.  # (bs, num_queries, 12)
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_angle`.
        """
        batch_size = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        pred_bboxes_list = [pred_bboxes[i] for i in range(batch_size)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               pred_bboxes_list,
                                               batch_gt_instances_3d,
                                               batch_input_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)  # (bs, 1, max_text_len 256)
        label_weights = torch.stack(label_weights_list, 0)  # (bs*num_query, 1)
        bbox_targets = torch.cat(bbox_targets_list, 0)  # (bs*num_query, 9)
        bbox_weights = torch.cat(bbox_weights_list, 0)  # (bs*num_query, 1)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert self.text_masks.dim() == 2
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(
            1)  # turn to bool and then (bs, 1, max_text_len)
        text_mask = text_mask.repeat(1, cls_scores.size(1),
                                     1)  # (bs, num_query, max_text_len)
        # cls_scores (bs, num_query, self.max_text_len 256)
        cls_scores = torch.masked_select(
            cls_scores, text_mask).contiguous()  # one-dimension

        labels = torch.masked_select(labels, text_mask)  # one-dimension
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        pred_bboxes = pred_bboxes.reshape(
            -1, pred_bboxes.size(-1))  # (bs*num_query, 12)

        valid_box_mask = bbox_weights[:, 0] > 0
        valid_bbox_preds = pred_bboxes[valid_box_mask]  # (bs, 9)
        valid_bbox_targets = bbox_targets[valid_box_mask]  # (bs, 9)

        if self.decouple_bbox_loss:
            bbox_targ_center = valid_bbox_targets[:, :3]
            bbox_targ_size = valid_bbox_targets[:, 3:6]
            bbox_targ_euler = valid_bbox_targets[:, 6:]
            bbox_pred_center = valid_bbox_preds[:, :3]
            bbox_pred_size = valid_bbox_preds[:, 3:6]
            bbox_pred_euler = valid_bbox_preds[:, 6:]

        corner_bbox_loss = 0
        if self.decouple_bbox_loss:
            assert self.decouple_groups in (
                3, 4), 'Only support groups=3 or 4 with stable performance.'
            if self.norm_decouple_loss:
                corner_bbox_loss += self.decouple_weights[0] * self.loss_bbox(
                    torch.concat(
                        (bbox_pred_center, bbox_targ_size, bbox_targ_euler),
                        dim=-1),
                    valid_bbox_targets,
                    reduction_override='none')
                corner_bbox_loss += self.decouple_weights[1] * self.loss_bbox(
                    torch.concat(
                        (bbox_targ_center, bbox_pred_size, bbox_targ_euler),
                        dim=-1),
                    valid_bbox_targets,
                    reduction_override='none')
                corner_bbox_loss += self.decouple_weights[2] * self.loss_bbox(
                    torch.concat(
                        (bbox_targ_center, bbox_targ_size, bbox_pred_euler),
                        dim=-1),
                    valid_bbox_targets,
                    reduction_override='none')
                bbox_sizes = bbox_targ_size.norm(dim=-1)[:,
                                                         None].clamp(min=0.1)
                corner_bbox_loss = (corner_bbox_loss / bbox_sizes).mean()
            else:
                corner_bbox_loss += self.decouple_weights[0] * self.loss_bbox(
                    torch.concat(
                        (bbox_pred_center, bbox_targ_size, bbox_targ_euler),
                        dim=-1), valid_bbox_targets)
                corner_bbox_loss += self.decouple_weights[1] * self.loss_bbox(
                    torch.concat(
                        (bbox_targ_center, bbox_pred_size, bbox_targ_euler),
                        dim=-1), valid_bbox_targets)
                corner_bbox_loss += self.decouple_weights[2] * self.loss_bbox(
                    torch.concat(
                        (bbox_targ_center, bbox_targ_size, bbox_pred_euler),
                        dim=-1), valid_bbox_targets)

            if self.decouple_groups == 4:
                corner_bbox_loss += self.decouple_weights[3] * self.loss_bbox(
                    valid_bbox_preds, valid_bbox_targets)

        else:
            corner_bbox_loss += self.loss_bbox(valid_bbox_preds,
                                               valid_bbox_targets)

        loss_bbox = corner_bbox_loss

        return loss_cls, loss_bbox


def normalize_vector(vector):
    norm = torch.norm(vector, dim=1, keepdim=True) + 1e-8
    normalized_vector = vector / norm
    return normalized_vector


def cross_product(a, b):
    cross_product = torch.cross(a, b, dim=1)
    return cross_product


def ortho_6d_2_Mat(x_raw, y_raw):
    """x_raw, y_raw: both tensors (batch, 3)."""
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)  # (batch, 3)
    x = cross_product(y, z)  # (batch, 3)

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x, y, z), 2)  # (batch, 3)
    return matrix
