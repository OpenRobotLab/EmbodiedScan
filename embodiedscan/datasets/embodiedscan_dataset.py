import os
import warnings
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from embodiedscan.registry import DATASETS
from embodiedscan.structures import get_box_type


@DATASETS.register_module()
class EmbodiedScanDataset(BaseDataset):
    r"""EmbodiedScan Dataset.

    This class serves as the API for experiments on the EmbodiedScan Dataset.

    Please refer to `EmbodiedScan Dataset
    <https://github.com/OpenRobotLab/EmbodiedScan>`_  for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        load_eval_anns (bool): Whether to load evaluation annotations.
            Defaults to True. Only take effect when test_mode is True.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        remove_dontcare (bool): Whether to remove objects that we do not care.
            Defaults to False.
        box_type_3d (str): To be deprecated?
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 box_type_3d: str = 'Euler-Depth',
                 **kwargs) -> None:

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.remove_dontcare = remove_dontcare
        self.load_eval_anns = load_eval_anns
        super().__init__(ann_file=ann_file,
                         metainfo=metainfo,
                         data_root=data_root,
                         pipeline=pipeline,
                         test_mode=test_mode,
                         **kwargs)

    def process_metainfo(self):
        """This function will be processed after metainfos from ann_file and
        config are combined."""
        assert 'categories' in self._metainfo

        if 'classes' not in self._metainfo:
            self._metainfo.setdefault(
                'classes', list(self._metainfo['categories'].keys()))

        self.label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        for key, value in self._metainfo['categories'].items():
            if key in self._metainfo['classes']:
                self.label_mapping[value] = self._metainfo['classes'].index(
                    key)

        self.occ_label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1,
            -1,
            dtype=int)
        if 'occ_classes' in self._metainfo:
            for idx, label_name in enumerate(self._metainfo['occ_classes']):
                self.occ_label_mapping[self.metainfo['categories'][
                    label_name]] = idx + 1  # 1-based, 0 is empty

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['box_type_3d'] = self.box_type_3d
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        info['scan_id'] = info['sample_idx']
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0
        # Because multi-view settings are different from original designs
        # we temporarily follow the ori design in ImVoxelNet
        info['img_path'] = []
        info['depth_img_path'] = []
        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info['images'])):
            img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                    info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.data_prefix.get('img_path', ''),
                                          info['images'][i]['depth_path'])

            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))
            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        info['depth2img'] = dict(extrinsic=extrinsics,
                                 intrinsic=cam2img,
                                 origin=np.array([.0, .0,
                                                  .5]).astype(np.float32))

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
            # Filter info with empty occupancy gts
            if self.filter_empty_gt:
                if 'gt_occupancy' in info['ann_info']:
                    if info['ann_info']['gt_occupancy'].shape[0] == 0:
                        return None
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = self._remove_dontcare(info['ann_info'])

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """

        ann_info = None
        if 'instances' in info and len(info['instances']) > 0:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((len(info['instances']), 9),
                                      dtype=np.float32),
                gt_labels_3d=np.zeros((len(info['instances']), ),
                                      dtype=np.int64),
            )
            for idx, instance in enumerate(info['instances']):
                ann_info['gt_bboxes_3d'][idx] = instance['bbox_3d']
                ann_info['gt_labels_3d'][idx] = self.label_mapping[
                    instance['bbox_label_3d']]

        # pack ann_info for return
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        # post-processing/filtering ann_info if not empty gt
        if 'visible_instance_ids' in info['images'][0]:
            ids = []
            for i in range(len(info['images'])):
                ids.append(info['images'][i]['visible_instance_ids'])
            mask_length = ann_info['gt_labels_3d'].shape[0]
            ann_info['visible_instance_masks'] = self._ids2masks(
                ids, mask_length)

        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)

        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'scannet':
            region = info['sample_idx'].split('/')[1]
            occ_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                        ann_dataset, 'scans', region,
                                        'occupancy', 'occupancy.npy')
            mask_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                         ann_dataset, 'scans', region,
                                         'occupancy', 'visible_occupancy.pkl')
        elif ann_dataset == '3rscan':
            region = info['sample_idx'].split('/')[1]
            occ_filename = os.path.join(self.data_prefix.get('img_path',
                                                             ''), ann_dataset,
                                        region, 'occupancy', 'occupancy.npy')
            mask_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                         ann_dataset, region, 'occupancy',
                                         'visible_occupancy.pkl')
        elif ann_dataset == 'matterport3d':
            building = info['sample_idx'].split('/')[1]
            region = info['sample_idx'].split('/')[2]
            occ_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                        ann_dataset, building, 'occupancy',
                                        f'occupancy_{region}.npy')
            mask_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                         ann_dataset, building, 'occupancy',
                                         f'visible_occupancy_{region}.pkl')
        else:
            raise NotImplementedError

        gt_occ = np.load(occ_filename)
        for i in range(gt_occ.shape[0]):
            cls_id = self.occ_label_mapping[gt_occ[i][3]]
            if cls_id < 0:
                cls_id = 255
            gt_occ[i][3] = cls_id
        ann_info['gt_occupancy'] = gt_occ

        ann_info['visible_occupancy_masks'] = []
        occ_masks = mmengine.load(mask_filename)
        for i in range(len(info['images'])):
            ann_info['visible_occupancy_masks'].append(
                occ_masks[i]['visible_occupancy'])

        ann_info['gt_bboxes_3d'] = self.box_type_3d(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        return ann_info

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _ids2masks(self, ids, mask_length):
        """Change visible_instance_ids to visible_instance_masks."""
        masks = []
        for idx in range(len(ids)):
            mask = np.zeros((mask_length, ), dtype=bool)
            mask[ids[idx]] = 1
            masks.append(mask)
        return masks

    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        -1 indicates dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            if key == 'instances':
                img_filtered_annotations[key] = ann_info[key]
            elif key == 'visible_instance_masks':
                img_filtered_annotations[key] = []
                for idx in range(len(ann_info[key])):
                    img_filtered_annotations[key].append(
                        ann_info[key][idx][filter_mask])
            elif key in ['gt_occupancy', 'visible_occupancy_masks']:
                pass
            else:
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
        return img_filtered_annotations

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        self.process_metainfo()

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif data_info is None:
                # When the data_info has empty anns, data_info=None
                pass
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list
