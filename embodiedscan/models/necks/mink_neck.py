# Copyright (c) OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py # noqa
from typing import List, Optional, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow get_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

import torch
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor, nn

from embodiedscan.registry import MODELS


@MODELS.register_module()
class MinkNeck(BaseModule):
    """MinkEngine based 3D sparse conv neck.

    Actually here we implement both the sparse 3D FPN and a head. The neck and
    the head can not be simply separated as pruning score on the i-th level
    of FPN requires classification scores from i+1-th level of the head.

    Args:
        num_classes (int): Number of classes.
        in_channels (tuple(int)): Number of channels in input tensors.
        out_channels (int): Number of channels in the neck output tensors.
        voxel_size (float): Voxel size in meters.
        pts_prune_threshold (int): Pruning threshold on each feature level.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(
            self,
            num_classes: int,  # 1
            in_channels: Tuple[int],
            out_channels: int,
            voxel_size: float,
            pts_prune_threshold: int,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None):
        super(MinkNeck, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, num_classes)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    @staticmethod
    def _make_up_block(in_channels: int, out_channels: int) -> nn.Module:
        """Construct DeConv-Norm-Act-Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels,
                                                       out_channels,
                                                       kernel_size=2,
                                                       stride=2,
                                                       dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    def _init_layers(self, in_channels: Tuple[int], out_channels: int,
                     num_classes: int):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
            out_channels (int): Number of channels in the neck output tensors.
            num_classes (int): Number of classes.
        """
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}',
                             self._make_block(in_channels[i], out_channels))

        # head layers
        self.conv_cls = ME.MinkowskiConvolution(out_channels,
                                                num_classes,
                                                kernel_size=1,
                                                bias=True,
                                                dimension=3)

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def forward(self, x: List[Tensor], batch_size) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        feats, cls_preds, points = [], [], []
        inputs = x
        x = inputs[-1]
        prune_score = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, prune_score)

            out = self.__getattr__(f'out_block_{i}')(x)
            feat, cls_pred, point, prune_score = \
                self._forward_single(out)
            feats.append(feat)
            cls_preds.append(cls_pred)
            points.append(point)
        batch_feats_list, batch_scores_list, batch_points_list = \
            self.convert_to_batch(feats, cls_preds, points, batch_size)
        return batch_feats_list, batch_scores_list, batch_points_list

    def _prune(self, x: SparseTensor, scores: SparseTensor) -> SparseTensor:
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def _forward_single(self, x: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass per level.

        Args:
            x (SparseTensor): Per level neck output tensor.

        Returns:
            tuple[Tensor]: Per level head predictions.
        """
        feat = x.features
        scores = self.conv_cls(x)
        cls_pred = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)

        feats, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            feats.append(feat[permutation])
            cls_preds.append(cls_pred[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return feats, cls_preds, points, prune_scores

    def convert_to_batch(self, feats, scores, points, batch_size):
        """Loss function about feature.

        Args:
            feats (list[list[Tensor]]): Feats for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            scores (list[list[Tensor]]): Scores for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            batch_size (int): Batch size.

        Returns:
            tuple[list[Tensor]]: Batch of features, scores and points lists.
        """
        batch_feats_list = []
        batch_scores_list = []
        batch_points_list = []
        for i in range(batch_size):
            feats_list = [x[i] for x in feats]
            scores_list = [x[i] for x in scores]
            points_list = [x[i] for x in points]
            batch_feats_list.append(torch.cat(feats_list, dim=0))
            batch_scores_list.append(torch.cat(scores_list, dim=0))
            batch_points_list.append(torch.cat(points_list, dim=0))

        return batch_feats_list, batch_scores_list, batch_points_list
