# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from typing import List, Union

import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from embodiedscan.registry import TASK_UTILS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth. This
    class computes an assignment between the targets and the predictions based
    on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched are
    treated as backgrounds. Thus each query prediction will be assigned with
    `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances_3d: InstanceData,
               gt_instances_3d: InstanceData,
               eps=1e-7) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs. This
        method assign each query prediction to a ground truth or background.
        The `assigned_gt_inds` with -1 means don't care, 0 means negative
        sample, and positive number is the index (1-based) of assigned gt.

        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances_3d (:obj:`InstanceData`): Predicted instances.
                It should includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_3d (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances_3d.labels_3d, Tensor)
        num_gts, num_preds = len(gt_instances_3d), len(pred_instances_3d)
        gt_labels = gt_instances_3d.labels_3d
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts=num_gts,
                                gt_inds=assigned_gt_inds,
                                max_overlaps=None,
                                labels=assigned_labels)

        # 2. compute the weighted costs
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(pred_instances=pred_instances_3d,
                              gt_instances=gt_instances_3d)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(num_gts=num_gts,
                            gt_inds=assigned_gt_inds,
                            max_overlaps=None,
                            labels=assigned_labels)
