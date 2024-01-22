from typing import Optional

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations

from embodiedscan.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """Load a depth image from file.

    Required Keys:

    - depth_img_path

    Modified Keys:

    - depth_img
    - depth_img_shape

    Args:
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.imdecode_backend = imdecode_backend

        self.backend_args = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['depth_img_path']
        depth_shift = results['depth_shift']

        try:
            depth_img_bytes = mmengine.fileio.get(
                filename, backend_args=self.backend_args)
            depth_img = mmcv.imfrombytes(depth_img_bytes,
                                         flag='unchanged',
                                         backend=self.imdecode_backend).astype(
                                             np.float32) / depth_shift
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['depth_img'] = depth_img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.backend_args is not None:
            repr_str += f'backend_args={self.backend_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


# TODO : refine
@TRANSFORMS.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - ann_info (dict)

        - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
          :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
          3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): Labels of ground truths.
          Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D ground truth bboxes.
          Only when `with_bbox` is True.
        - gt_labels (np.ndarray): Labels of ground truths.
          Only when `with_label` is True.
        - depths (np.ndarray): Only when
          `with_bbox_depth` is True.
        - centers_2d (np.ndarray): Only when
          `with_bbox_depth` is True.
        - attr_labels (np.ndarray): Attribute labels of instances.
          Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): Path of instance mask file.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): Path of semantic mask file.
      Only when `with_seg_3d` is True.
    - pts_panoptic_mask_path (str): Path of panoptic mask file.
      Only when both `with_panoptic_3d` is True.

    Added Keys:

    - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
      :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
      3D ground truth bboxes. Only when `with_bbox_3d` is True
    - gt_labels_3d (np.int64): Labels of ground truths.
      Only when `with_label_3d` is True.
    - gt_bboxes (np.float32): 2D ground truth bboxes.
      Only when `with_bbox` is True.
    - gt_labels (np.int64): Labels of ground truths.
      Only when `with_label` is True.
    - depths (np.float32): Only when
      `with_bbox_depth` is True.
    - centers_2d (np.ndarray): Only when
      `with_bbox_depth` is True.
    - attr_labels (np.int64): Attribute labels of instances.
      Only when `with_attr_label` is True.
    - pts_instance_mask (np.int64): Instance mask of each point.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask (np.int64): Semantic mask of each point.
      Only when `with_seg_3d` is True.

    Args:
        with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool): Whether to load 3D instance masks for points.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D semantic masks for points.
            Defaults to False.
        with_bbox (bool): Whether to load 2D boxes. Defaults to False.
        with_label (bool): Whether to load 2D labels. Defaults to False.
        with_mask (bool): Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
        with_panoptic_3d (bool): Whether to load 3D panoptic masks for points.
            Defaults to False.
        poly2mask (bool): Whether to convert polygon annotations to bitmasks.
            Defaults to True.
        seg_3d_dtype (str): String of dtype of 3D semantic masks.
            Defaults to 'np.int64'.
        seg_offset (int): The offset to split semantic and instance labels from
            panoptic labels. Defaults to None.
        dataset_type (str): Type of dataset used for splitting semantic and
            instance labels. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 with_bbox_3d: bool = True,
                 with_label_3d: bool = True,
                 with_depth_map: bool = False,
                 with_attr_label: bool = False,
                 with_mask_3d: bool = False,
                 with_seg_3d: bool = False,
                 with_bbox: bool = False,
                 with_label: bool = False,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 with_bbox_depth: bool = False,
                 with_panoptic_3d: bool = False,
                 with_visible_instance_masks: bool = False,
                 with_occupancy: bool = False,
                 with_visible_occupancy_masks: bool = False,
                 poly2mask: bool = True,
                 seg_3d_dtype: str = 'np.int64',
                 seg_offset: int = None,
                 dataset_type: str = None,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__(with_bbox=with_bbox,
                         with_label=with_label,
                         with_mask=with_mask,
                         with_seg=with_seg,
                         poly2mask=poly2mask,
                         backend_args=backend_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_depth_map = with_depth_map
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_panoptic_3d = with_panoptic_3d
        self.with_visible_instance_masks = with_visible_instance_masks
        self.with_occupancy = with_occupancy
        self.with_visible_occupancy_masks = with_visible_occupancy_masks
        self.seg_3d_dtype = eval(seg_3d_dtype)
        self.seg_offset = seg_offset
        self.dataset_type = dataset_type

    def _load_bboxes_3d(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_depth_map(self, results: dict) -> dict:

        img_filename = results['img_path']
        pts_filename = img_filename.replace('samples', 'depth_points') + '.bin'
        results['depth_map_path'] = pts_filename
        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.backend_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        pts_img = points.reshape(-1, 3)
        img_shape = results['ori_shape']
        depth_img = np.zeros(img_shape, dtype=np.float32)
        iy = np.round(pts_img[:, 1]).astype(np.int64)
        ix = np.round(pts_img[:, 0]).astype(np.int64)
        depth_img[iy, ix] = pts_img[:, 2]
        results['depth_map'] = depth_img

        return results

    def _load_masks_3d(self, results: dict) -> dict:
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['pts_instance_mask_path']

        try:
            mask_bytes = mmengine.fileio.get(pts_instance_mask_path,
                                             backend_args=self.backend_args)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(pts_instance_mask_path,
                                            dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        try:
            mask_bytes = mmengine.fileio.get(pts_semantic_mask_path,
                                             backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(mask_bytes,
                                              dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(pts_semantic_mask_path,
                                            dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes loads semantic and panoptic labels from different files.

        results['pts_semantic_mask'] = pts_semantic_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results

    def _load_panoptic_3d(self, results: dict) -> dict:
        """Private function to load 3D panoptic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the panoptic segmentation annotations.
        """
        pts_panoptic_mask_path = results['pts_panoptic_mask_path']

        try:
            mask_bytes = mmengine.fileio.get(pts_panoptic_mask_path,
                                             backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_panoptic_mask = np.frombuffer(mask_bytes,
                                              dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_panoptic_mask_path)
            pts_panoptic_mask = np.fromfile(pts_panoptic_mask_path,
                                            dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        elif self.dataset_type == 'nuscenes':
            pts_semantic_mask = pts_semantic_mask // self.seg_offset

        results['pts_semantic_mask'] = pts_semantic_mask

        # We can directly take panoptic labels as instance ids.
        pts_instance_mask = pts_panoptic_mask.astype(np.int64)
        results['pts_instance_mask'] = pts_instance_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_visible_instance_masks(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['visible_instance_masks'] = results['ann_info'][
            'visible_instance_masks']
        return results

    def _load_occupancy(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['gt_occupancy'] = results['ann_info']['gt_occupancy']
        return results

    def _load_visible_occupancy_masks(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        results['visible_occupancy_masks'] = results['ann_info'][
            'visible_occupancy_masks']
        return results

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_depth_map:
            results = self._load_depth_map(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)
        if self.with_visible_instance_masks:
            results = self._load_visible_instance_masks(results)
        if self.with_occupancy:
            results = self._load_occupancy(results)
        if self.with_visible_occupancy_masks:
            results = self._load_visible_occupancy_masks(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_panoptic_3d={self.with_panoptic_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask}), '
        repr_str += f'{indent_str}seg_offset={self.seg_offset}), '
        repr_str += f'{indent_str}with_visible_instance_masks='
        repr_str += f'{self.with_visible_instance_masks}), '
        repr_str += f'{indent_str}with_occupancy={self.with_occupancy}), '
        repr_str += f'{indent_str}with_visible_occupancy_masks='
        repr_str += f'{self.with_visible_occupancy_masks})'

        return repr_str
