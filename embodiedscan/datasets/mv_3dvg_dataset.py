# Copyright (c) OpenRobotLab. All rights reserved.
import os
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from embodiedscan.registry import DATASETS
from embodiedscan.structures import get_box_type


@DATASETS.register_module()
class MultiView3DGroundingDataset(BaseDataset):
    r"""Multi-View 3D Grounding Dataset for EmbodiedScan.

    This class serves as the API for experiments on the EmbodiedScan Dataset.

    Please refer to `EmbodiedScan Dataset
    <https://github.com/OpenRobotLab/EmbodiedScan>`_  for data downloading.

    TODO: Merge the implementation with EmbodiedScanDataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        vg_file (str): Path of the visual grounding annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Euler-Depth' in this dataset.
        serialize_data (bool): Whether to serialize all data samples to save
            memory. Defaults to False. It is set to True typically, but we
            need to do the serialization after getting the data_list through
            the preliminary loading and converting. Therefore, we set it to
            False by default and serialize data samples at last meanwhile
            setting this attribute to True.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        remove_dontcare (bool): Whether to remove objects that we do not care.
            Defaults to False.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        load_eval_anns (bool): Whether to load evaluation annotations.
            Defaults to True. Only take effect when test_mode is True.
    """
    # NOTE: category "step" -> "steps" to avoid potential naming conflicts in
    # TensorboardVisBackend
    METAINFO = {
        'classes':
        ('adhesive tape', 'air conditioner', 'alarm', 'album', 'arch',
         'backpack', 'bag', 'balcony', 'ball', 'banister', 'bar', 'barricade',
         'baseboard', 'basin', 'basket', 'bathtub', 'beam', 'beanbag', 'bed',
         'bench', 'bicycle', 'bidet', 'bin', 'blackboard', 'blanket', 'blinds',
         'board', 'body loofah', 'book', 'boots', 'bottle', 'bowl', 'box',
         'bread', 'broom', 'brush', 'bucket', 'cabinet', 'calendar', 'camera',
         'can', 'candle', 'candlestick', 'cap', 'car', 'carpet', 'cart',
         'case', 'ceiling', 'chair', 'chandelier', 'cleanser', 'clock',
         'clothes', 'clothes dryer', 'coat hanger', 'coffee maker', 'coil',
         'column', 'commode', 'computer', 'conducting wire', 'container',
         'control', 'copier', 'cosmetics', 'couch', 'counter', 'countertop',
         'crate', 'crib', 'cube', 'cup', 'curtain', 'cushion', 'decoration',
         'desk', 'detergent', 'device', 'dish rack', 'dishwasher', 'dispenser',
         'divider', 'door', 'door knob', 'doorframe', 'doorway', 'drawer',
         'dress', 'dresser', 'drum', 'duct', 'dumbbell', 'dustpan', 'dvd',
         'eraser', 'excercise equipment', 'fan', 'faucet', 'fence', 'file',
         'fire extinguisher', 'fireplace', 'floor', 'flowerpot', 'flush',
         'folder', 'food', 'footstool', 'frame', 'fruit', 'furniture',
         'garage door', 'garbage', 'glass', 'globe', 'glove', 'grab bar',
         'grass', 'guitar', 'hair dryer', 'hamper', 'handle', 'hanger', 'hat',
         'headboard', 'headphones', 'heater', 'helmets', 'holder', 'hook',
         'humidifier', 'ironware', 'jacket', 'jalousie', 'jar', 'kettle',
         'keyboard', 'kitchen island', 'kitchenware', 'knife', 'label',
         'ladder', 'lamp', 'laptop', 'ledge', 'letter', 'light', 'luggage',
         'machine', 'magazine', 'mailbox', 'map', 'mask', 'mat', 'mattress',
         'menu', 'microwave', 'mirror', 'molding', 'monitor', 'mop', 'mouse',
         'napkins', 'notebook', 'object', 'ottoman', 'oven', 'pack', 'package',
         'pad', 'pan', 'panel', 'paper', 'paper cutter', 'partition',
         'pedestal', 'pen', 'person', 'piano', 'picture', 'pillar', 'pillow',
         'pipe', 'pitcher', 'plant', 'plate', 'player', 'plug', 'plunger',
         'pool', 'pool table', 'poster', 'pot', 'price tag', 'printer',
         'projector', 'purse', 'rack', 'radiator', 'radio', 'rail',
         'range hood', 'refrigerator', 'remote control', 'ridge', 'rod',
         'roll', 'roof', 'rope', 'sack', 'salt', 'scale', 'scissors', 'screen',
         'seasoning', 'shampoo', 'sheet', 'shelf', 'shirt', 'shoe', 'shovel',
         'shower', 'sign', 'sink', 'soap', 'soap dish', 'soap dispenser',
         'socket', 'speaker', 'sponge', 'spoon', 'stairs', 'stall', 'stand',
         'stapler', 'statue', 'steps', 'stick', 'stool', 'stopcock', 'stove',
         'structure', 'sunglasses', 'support', 'switch', 'table', 'tablet',
         'teapot', 'telephone', 'thermostat', 'tissue', 'tissue box',
         'toaster', 'toilet', 'toilet paper', 'toiletry', 'tool', 'toothbrush',
         'toothpaste', 'towel', 'toy', 'tray', 'treadmill', 'trophy', 'tube',
         'tv', 'umbrella', 'urn', 'utensil', 'vacuum cleaner', 'vanity',
         'vase', 'vent', 'ventilation', 'wall', 'wardrobe', 'washbasin',
         'washing machine', 'water cooler', 'water heater', 'window',
         'window frame', 'windowsill', 'wine', 'wire', 'wood', 'wrap'),
        'valid_class_ids':
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
         55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
         89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
         105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
         133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
         147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
         161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
         175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
         189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
         203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
         217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
         231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
         245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
         259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
         273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
         287, 288)
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 vg_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'Euler-Depth',
                 serialize_data: bool = False,
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 **kwargs) -> None:

        if 'classes' in metainfo:
            if metainfo['classes'] == 'all':
                metainfo['classes'] = list(self.METAINFO['classes'])

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.remove_dontcare = remove_dontcare
        self.load_eval_anns = load_eval_anns

        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         metainfo=metainfo,
                         pipeline=pipeline,
                         serialize_data=serialize_data,
                         test_mode=test_mode,
                         **kwargs)

        self.vg_file = osp.join(self.data_root, vg_file)
        self.convert_info_to_scan()
        self.data_list = self.load_language_data()
        self.data_bytes, self.data_address = self._serialize_data()
        self.serialize_data = True

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

    # need to compensate the scan_id info to the original pkl file
    def convert_info_to_scan(self):
        self.scans = dict()
        for data in self.data_list:
            scan_id = data['scan_id']
            self.scans[scan_id] = data

    @staticmethod
    def _is_view_dep(text):
        """Check whether to augment based on sr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing', 'leftmost',
            'rightmost', 'looking', 'across'
        ]
        words = set(text.split())
        return any(rel in words for rel in rels)

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

    def load_language_data(self):
        # load the object-level annotations
        language_annotations = load(self.vg_file)
        # language_infos = [
        #     {
        #         'scan_id': anno['scan_id'],
        #         'text': anno['text'],
        #         'target_id': int(anno['target_id']), (training)
        #         'distractor_ids': anno['distractor_ids'], (training)
        #         'tokens_positive': anno['tokens_positive'] (training)
        #     }
        #     for anno in language_annotations
        # ]
        # According to each object annotation,
        # find all objects in the corresponding scan
        language_infos = []
        for anno in mmengine.track_iter_progress(language_annotations):
            language_info = dict()
            language_info.update({
                'scan_id': anno['scan_id'],
                'text': anno['text']
            })
            data = self.scans[language_info['scan_id']]
            language_info['axis_align_matrix'] = data['axis_align_matrix']
            language_info['img_path'] = data['img_path']
            language_info['depth_img_path'] = data['depth_img_path']
            language_info['depth2img'] = data['depth2img']
            if 'cam2img' in data:
                language_info['cam2img'] = data['cam2img']
            language_info['scan_id'] = data['scan_id']
            language_info['depth_shift'] = data['depth_shift']
            language_info['depth_cam2img'] = data['depth_cam2img']

            ann_info = data['ann_info']

            # save the bounding boxes and corresponding labels
            language_anno_info = dict()
            language_anno_info['is_view_dep'] = self._is_view_dep(
                language_info['text'])
            labels = ann_info['gt_labels_3d']  # all box labels in the scan
            bboxes = ann_info['gt_bboxes_3d']  # BaseInstanceBboxes
            if 'target_id' in anno:  # w/ ground truths
                language_info.update({'target_id': int(anno['target_id'])})
                # obtain all objects sharing the same category with
                # the target object, the num of such objects <= 32
                object_ids = ann_info['bbox_id']  # numpy array
                object_ind = np.where(
                    object_ids == language_info['target_id'])[0]
                if len(object_ind) != 1:
                    continue
                language_anno_info['gt_bboxes_3d'] = bboxes[object_ind]
                language_anno_info['gt_labels_3d'] = labels[object_ind]
                # include other optional keys
                optional_keys = ['distractor_ids', 'tokens_positive']
                for key in optional_keys:
                    if key in anno:
                        language_info.update({key: anno[key]})
                # the 'distractor_ids' starts from 1, not 0
                language_anno_info['is_hard'] = len(
                    language_info['distractor_ids']
                ) > 3  # more than three distractors
                language_anno_info['is_unique'] = len(
                    language_info['distractor_ids']) == 0
            else:
                # inference w/o gt, assign the placeholder gt_boxes and labels
                language_anno_info['gt_bboxes_3d'] = bboxes
                language_anno_info['gt_labels_3d'] = labels
                # placeholder value for 'is_hard' and 'is_unique'
                language_anno_info['is_hard'] = False
                language_anno_info['is_unique'] = False

            if not self.test_mode:
                language_info['ann_info'] = language_anno_info

            if self.test_mode and self.load_eval_anns:
                language_info['ann_info'] = language_anno_info
                language_info['eval_ann_info'] = language_info['ann_info']

            language_infos.append(language_info)

        del self.scans

        return language_infos

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
        # Because multi-view settings are different from original designs
        # we temporarily follow the ori design in ImVoxelNet
        info['img_path'] = []
        info['depth_img_path'] = []
        info['scan_id'] = info['sample_idx']
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0

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
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = info['ann_info']
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
            # add s or gt prefix for most keys after concat
            # we only process 3d annotations here, the corresponding
            # 2d annotation process is in the `LoadAnnotations3D`
            # in `transforms`
            name_mapping = {
                'bbox_label_3d': 'gt_labels_3d',
                'bbox_label': 'gt_bboxes_labels',
                'bbox': 'gt_bboxes',
                'bbox_3d': 'gt_bboxes_3d',
                'depth': 'depths',
                'center_2d': 'centers_2d',
                'attr_label': 'attr_labels',
                'velocity': 'velocities',
            }
            instances = info['instances']
            # empty gt
            if len(instances) == 0:
                return None
            else:
                keys = list(instances[0].keys())
                ann_info = dict()
                for ann_name in keys:
                    temp_anns = [item[ann_name] for item in instances]
                    # map the original dataset label to training label
                    if 'label' in ann_name and ann_name != 'attr_label':
                        temp_anns = [
                            self.label_mapping[item] for item in temp_anns
                        ]
                    if ann_name in name_mapping:
                        mapped_ann_name = name_mapping[ann_name]
                    else:
                        mapped_ann_name = ann_name

                    if 'label' in ann_name:
                        temp_anns = np.array(temp_anns).astype(np.int64)
                    elif ann_name in name_mapping:
                        temp_anns = np.array(temp_anns).astype(np.float32)
                    else:
                        temp_anns = np.array(temp_anns)

                    ann_info[mapped_ann_name] = temp_anns
                ann_info['instances'] = info['instances']

        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        ann_info['gt_bboxes_3d'] = self.box_type_3d(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5))

        return ann_info
