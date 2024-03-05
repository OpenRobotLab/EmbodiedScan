from abc import abstractmethod
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from embodiedscan.registry import TASK_UTILS
from embodiedscan.structures import EulerDepthInstance3DBoxes


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


@TASK_UTILS.register_module()
class BBox3DL1Cost(BaseMatchCost):
    """L1 cost for 3D boxes."""

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
                which are all in range [0, 1] and shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth boxes with `normalized`
                coordinates (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y).
                Shape [num_gt, 10].
        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes_3d.tensor  # (num_preds, 9)
        gt_bboxes = gt_instances.bboxes_3d.tensor  # (num_gts, 9)

        bbox_cost = torch.cdist(pred_bboxes, gt_bboxes,
                                p=1)  # (num_preds, num_gt)
        return bbox_cost * self.weight


@TASK_UTILS.register_module()
class TokenMapCost(BaseMatchCost):
    """TokenPredictionCost."""

    def __call__(self, pred_logits: Tensor, gt_logits: Tensor) -> Tensor:
        """Compute match cost.

        Args:
            pred_logits (Tensor): Shape [num_query, C].
            gt_logits (Tensor): Shape [num_gt, C].
        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        token_map_cost = torch.matmul(pred_logits, gt_logits.transpose(0, 1))
        return token_map_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost(object):
    """3D IoU cost for 3D boxes."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData):
        pred_bboxes = EulerDepthInstance3DBoxes(
            pred_instances.bboxes_3d.tensor, origin=(0.5, 0.5, 0.5))
        gt_bboxes = EulerDepthInstance3DBoxes(gt_instances.bboxes_3d.tensor,
                                              origin=(0.5, 0.5, 0.5))
        overlaps = pred_bboxes.overlaps(pred_bboxes, gt_bboxes)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps

        return iou_cost * self.weight


@TASK_UTILS.register_module()
class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 binary_input: bool = False,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
                in shape (num_queries, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_queries, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if self.binary_input:
            pred_masks = pred_instances.masks
            gt_masks = gt_instances.masks
            return self._mask_focal_loss_cost(pred_masks, gt_masks)
        else:
            pred_scores = pred_instances.scores
            gt_labels = gt_instances.labels
            return self._focal_loss_cost(pred_scores, gt_labels)


@TASK_UTILS.register_module()
class BinaryFocalLossCost(FocalLossCost):
    """Binary focal loss cost."""

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # gt_instances.text_token_mask is a repeated tensor of the same length
        # of instances. Only gt_instances.text_token_mask[0] is useful
        text_token_mask = torch.nonzero(
            gt_instances.text_token_mask[0]).squeeze(-1)
        # mask used to filter padding texts
        # (num_query,)
        pred_scores = pred_instances.scores_3d[:, text_token_mask]
        # (1, real_tex_length)
        gt_labels = gt_instances.positive_maps[:, text_token_mask]
        return self._focal_loss_cost(pred_scores, gt_labels)
