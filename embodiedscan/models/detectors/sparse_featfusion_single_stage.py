# Copyright (c) OpenRobotLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = None
    pass

from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from embodiedscan.registry import MODELS
from embodiedscan.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedscan.utils import ConfigType
from embodiedscan.utils.typing_config import (ForwardResults, InstanceList,
                                              SampleList)

from ..layers.fusion_layers.point_fusion import (batch_point_sample,
                                                 point_sample)


@MODELS.register_module()
class SparseFeatureFusionSingleStage3DDetector(BaseModel):
    """SparseFusionSingleStage3DDetector.

    Args:
        backbone (dict): Config dict of detector's backbone.
        backbone_3d (dict): Config dict of detector's 3D backbone.
        bbox_head (dict): Config dict of box head.
        neck (dict, optional): Config dict of neck. Defaults to None.
        neck_3d (dict, optional): Config dict of 3D neck. Defaults to None.
        coord_type (str): Type of Box coordinates. Defaults to CAMERA.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        use_xyz_feat (bool): Whether to use xyz features.
            Defaults to False.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    _version = 2

    def __init__(self,
                 backbone: ConfigType,
                 backbone_3d: ConfigType,
                 bbox_head: ConfigType,
                 neck: ConfigType = None,
                 neck_3d: ConfigType = None,
                 coord_type: str = 'CAMERA',
                 train_cfg: Optional[ConfigType] = None,
                 test_cfg: Optional[ConfigType] = None,
                 data_preprocessor: Optional[ConfigType] = None,
                 use_xyz_feat: bool = False,
                 init_cfg: Optional[ConfigType] = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone_3d = MODELS.build(backbone_3d)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if neck_3d is not None:
            self.neck_3d = MODELS.build(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )
        self.voxel_size = bbox_head['voxel_size']
        self.use_xyz_feat = use_xyz_feat

    def extract_feat(
        self, batch_inputs_dict: Dict[str,
                                      Tensor], batch_data_samples: SampleList
    ) -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which includes
                'points' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']

        # construct sparse tensor and features
        # coordinates shape: (N, D+1), features shape: (N, F)
        # N is the total point number in the batch
        if self.use_xyz_feat:
            coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p) for p in points],
                device=points[0].device)
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
                device=points[0].device)

        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone_3d(x)
        num_levels = len(x)  # 4 levels
        num_samples = len(x[0].decomposed_coordinates)

        # extract img features
        img = batch_inputs_dict['imgs']
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]

        if len(img.shape) > 4:  # (B, n_views, C, H, W)
            img = img.reshape([-1] + list(img.shape)[2:])
            img_features = self.backbone(img)
            img_features = [
                img_feat.reshape([batch_size, -1] + list(img_feat.shape)[1:])
                for img_feat in img_features
            ]
        else:
            img_features = self.backbone(img)

        all_points_imgfeats = []

        for idx in range(len(batch_img_metas)):
            img_meta = batch_img_metas[idx]
            img_scale_factor = (img.new_tensor(img_meta['scale_factor'][:2])
                                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (img.new_tensor(img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View Sparse Fusion
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []
                # Support different intrinsic matrices for different images
                # if the original intrinsic is only a matrix
                # we will simply copy it to construct the intrinsic matrix list
                # in MultiViewPipeline
                assert isinstance(proj_mat['intrinsic'], list)
                for proj_idx in range(len(proj_mat['extrinsic'])):
                    intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                    extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                proj_mat = torch.stack(projection)
                points_imgfeats = []
                for level_idx in range(num_levels):
                    point = x[level_idx].decomposed_coordinates[
                        idx] * self.voxel_size
                    points_imgfeat = batch_point_sample(
                        img_meta,
                        img_features=img_features[level_idx][idx],
                        points=point,
                        proj_mat=proj_mat,
                        coord_type=self.coord_type,
                        img_scale_factor=img_scale_factor,
                        img_crop_offset=img_crop_offset,
                        img_flip=img_flip,
                        img_pad_shape=img.shape[-2:],
                        img_shape=img_meta['img_shape'][:2],
                        aligned=False)
                    points_imgfeats.append(
                        points_imgfeat)  # one sample, all levels
            else:
                feature = img_features[idx]
                proj_mat = points.new_tensor(proj_mat)
                points_imgfeats = []
                for level_idx in range(num_levels):
                    # get the corresponding voxel coordinates
                    # and * voxel_size to get the absolute positions
                    point = x[level_idx].decomposed_coordinates[
                        idx] * self.voxel_size
                    points_imgfeat = point_sample(
                        img_meta,
                        img_features=feature[None, ...],
                        points=point,
                        proj_mat=point.new_tensor(proj_mat),
                        coord_type='CAMERA',
                        img_scale_factor=img_scale_factor,
                        img_crop_offset=img_crop_offset,
                        img_flip=img_flip,
                        img_pad_shape=img.shape[-2:],
                        img_shape=img_meta['img_shape'][:2],
                        aligned=False)
                    points_imgfeats.append(
                        points_imgfeat)  # one sample, all levels
            all_points_imgfeats.append(
                points_imgfeats)  # all samples, all levels

        # append img features
        for level_idx in range(num_levels):
            mlvl_feats = torch.cat([
                all_points_imgfeats[sample_idx][level_idx]
                for sample_idx in range(num_samples)
            ])
            img_x = ME.SparseTensor(
                features=mlvl_feats,
                coordinate_map_key=x[level_idx].coordinate_map_key,
                coordinate_manager=x[level_idx].coordinate_manager)
            x[level_idx] = ME.cat(x[level_idx], img_x)

        return x

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: Optional[SampleList] = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results = self.bbox_head.forward(x)
        return results

    @property
    def with_neck(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_3d(self):
        """Whether the detector has a 3D neck."""
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
