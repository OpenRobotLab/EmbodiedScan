from typing import List, Union

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import RandomFlip

from embodiedscan.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        sync_2d (bool): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_2d (bool): Whether to apply flip for the img data.
            If True, it will adopt the flip augmentation for the img.
            False occurs on bev augmentation for bev-based image 3d det.
            Defaults to True.
        flip_3d (bool): Whether to apply flip for the 3d point cloud data.
            If True, it will adopt the flip augmentation for the point cloud.
            Defaults to True.
        flip_ratio_bev_horizontal (float): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float): The flipping probability
            in vertical direction. Defaults to 0.0.
        flip_box3d (bool): Whether to flip bounding box. In most of the case,
            the box should be fliped. In cam-based bev detection, this is set
            to False, since the flip of 2D images does not influence the 3D
            box. Defaults to True.
    """

    def __init__(self,
                 sync_2d: bool = True,
                 flip_2d: bool = True,
                 flip_3d: bool = True,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0,
                 flip_box3d: bool = True,
                 update_lidar2cam: bool = False,
                 **kwargs) -> None:
        # `flip_ratio_bev_horizontal` is equal to
        # for flip prob of 2d image when
        # `sync_2d` is True
        super(RandomFlip3D, self).__init__(prob=flip_ratio_bev_horizontal,
                                           direction='horizontal',
                                           **kwargs)
        self.sync_2d = sync_2d
        self.flip_2d = flip_2d
        self.flip_3d = flip_3d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_box3d = flip_box3d
        self.update_lidar2cam = update_lidar2cam
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(flip_ratio_bev_horizontal, (int, float)) \
                    and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(flip_ratio_bev_vertical, (int, float)) \
                    and 0 <= flip_ratio_bev_vertical <= 1

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        # flip 2D image and its annotations
        if self.flip_2d:
            # only handle the 2D image
            if 'img' in input_dict:
                super(RandomFlip3D, self).transform(input_dict)
            flip = input_dict.get('flip', False)
            if flip:
                input_dict = self.random_flip_data_2d(input_dict)

        if self.flip_3d:
            # only handle the 3D points
            if self.sync_2d and 'img' in input_dict:
                # TODO check if this is necessary in FOCS3D
                input_dict['pcd_horizontal_flip'] = input_dict['flip']
                input_dict['pcd_vertical_flip'] = False
            else:
                if 'pcd_horizontal_flip' not in input_dict:
                    if np.random.rand() < self.flip_ratio_bev_horizontal:
                        flip_horizontal = True
                    else:
                        flip_horizontal = False
                    input_dict['pcd_horizontal_flip'] = flip_horizontal
                if 'pcd_vertical_flip' not in input_dict:
                    if np.random.rand() < self.flip_ratio_bev_vertical:
                        flip_vertical = True
                    else:
                        flip_vertical = False
                    input_dict['pcd_vertical_flip'] = flip_vertical

            if 'transformation_3d_flow' not in input_dict:
                input_dict['transformation_3d_flow'] = []

            if input_dict['pcd_horizontal_flip']:
                self.random_flip_data_3d(input_dict, 'horizontal')
                input_dict['transformation_3d_flow'].extend(['HF'])
            if input_dict['pcd_vertical_flip']:
                self.random_flip_data_3d(input_dict, 'vertical')
                input_dict['transformation_3d_flow'].extend(['VF'])
            if self.update_lidar2cam:
                self._transform_lidar2cam(input_dict)
        return input_dict

    def random_flip_data_3d(self,
                            input_dict: dict,
                            direction: str = 'horizontal') -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if self.flip_box3d:
            if 'gt_bboxes_3d' in input_dict:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict['gt_bboxes_3d'].flip(
                        direction, points=input_dict['points'])
                else:
                    # vision-only detection
                    input_dict['gt_bboxes_3d'].flip(direction)
            else:
                input_dict['points'].flip(direction)

    def random_flip_data_2d(self,
                            input_dict: dict,
                            direction: str = 'horizontal') -> dict:
        if 'centers_2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers_2d'][..., 0] = \
                w - input_dict['centers_2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

        if 'fov_ori2aug' not in input_dict:
            fov_ori2aug = np.eye(4, 4)
        else:
            fov_ori2aug = input_dict['fov_ori2aug']
        # get the value of w
        w = input_dict['img_shape'][1]
        # flip_matrix[0,0] = -1
        # flip_matrix[0,3] = w
        # fov_ori2aug = np.matmul(fov_ori2aug, flip_matrix)
        fov_ori2aug[0] *= -1
        fov_ori2aug[0, 3] += w
        input_dict['fov_ori2aug'] = fov_ori2aug
        return input_dict

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints.

        Add the override feature that if 'flip' is already in results, use it
        to do the augmentation.
        """
        if 'flip' not in results:
            cur_dir = self._choose_direction()
        else:
            # `flip_direction` works only when `flip` is True.
            # For example, in `MultiScaleFlipAug3D`, `flip_direction` is
            # 'horizontal' but `flip` is False.
            if results['flip']:
                assert 'flip_direction' in results, 'flip and flip_direction '
                'must exist simultaneously'
                cur_dir = results['flip_direction']
            else:
                cur_dir = None
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def _transform_lidar2cam(self, results: dict) -> None:
        """TODO."""
        aug_matrix = np.eye(4)
        if results.get('pcd_horizontal_flip', False):
            aug_matrix[1, 1] *= -1
        if results.get('pcd_vertical_flip', False):
            aug_matrix[0, 0] *= -1
        lidar2cam_list = []
        for lidar2cam in results['lidar2cam']:
            lidar2cam = np.array(lidar2cam)
            lidar2cam = np.matmul(lidar2cam, aug_matrix)
            lidar2cam_list.append(lidar2cam.tolist())
        results['lidar2cam'] = lidar2cam_list

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@TRANSFORMS.register_module()
class GlobalRotScaleTrans(BaseTransform):
    """Apply global rotation, scaling and translation to a 3D scene.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        rot_dof (int): DoF of rotation noise. Defaults to 1.
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range: Union[List[float], int,
                                  float] = [-0.78539816, 0.78539816],
                 rot_dof: int = 1,
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False,
                 update_lidar2cam: bool = False) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range
        self.rot_dof = rot_dof
        self.update_lidar2cam = update_lidar2cam

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        if self.update_lidar2cam:
            self._transform_lidar2cam(input_dict)
        return input_dict

    def _trans_bbox_points(self, input_dict: dict) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        if 'points' in input_dict:
            input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)

    def _rot_bbox_points(self, input_dict: dict) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        rotation = self.rot_range
        if self.rot_dof == 1:
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
            noise_rotation *= -1
        elif self.rot_dof > 1:
            noise_rotation = np.array([
                -np.random.uniform(rotation[0], rotation[1]),
                -np.random.uniform(rotation[0], rotation[1]),
                -np.random.uniform(rotation[0], rotation[1])
            ])
        else:
            raise NotImplementedError
        # TODO delete this. And -1 is to align the rotation with
        # the version of 0.17.
        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            # rotate points with bboxes
            if 'points' in input_dict:
                points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
            else:
                rot_mat_T = input_dict['gt_bboxes_3d'].rotate(noise_rotation)
        elif 'points' in input_dict:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict['points'].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        if 'points' in input_dict:
            points = input_dict['points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                  'setting shift_height=True \
                   but points have no height attribute'

                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['points'] = points

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)

    def _random_scale(self, input_dict: dict) -> None:
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor'
            are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def _transform_lidar2cam(self, input_dict: dict) -> None:
        aug_matrix = np.eye(4)

        if 'pcd_rotation' in input_dict:
            aug_matrix[:3, :3] = input_dict['pcd_rotation'].T.numpy(
            ) * input_dict['pcd_scale_factor']
        else:
            aug_matrix[:3, :3] = np.eye(3).view(
                1, 3, 3) * input_dict['pcd_scale_factor']
        aug_matrix[:3, -1] = input_dict['pcd_trans'].reshape(1, 3)
        aug_matrix[-1, -1] = 1.0
        aug_matrix = np.linalg.inv(aug_matrix)
        lidar2cam_list = []
        for lidar2cam in input_dict['lidar2cam']:
            lidar2cam = np.array(lidar2cam)
            lidar2cam = np.matmul(lidar2cam, aug_matrix)
            lidar2cam_list.append(lidar2cam.tolist())
        input_dict['lidar2cam'] = lidar2cam_list

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str
