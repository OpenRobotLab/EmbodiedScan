import json
import os
import os.path as osp
import pickle
import random
from copy import deepcopy
from glob import glob
from typing import Dict, List

import numpy as np
import torch
import utils.pc_util as pc_util
from datasets.scannet_base_dataset import (BASE, DatasetConfig,
                                           ScanNetBaseDataset)
from datasets.task_prompts import BOX_FORMAT, TASK_PROPMT
from eval_utils.evaluate_mmscan import evaluate
from transformers import AutoTokenizer

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points

from mmscan import MMScan


class Dataset(ScanNetBaseDataset):

    def __init__(
        self,
        args,
        dataset_config,
        split_set='train',
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )

        assert split_set in ['train', 'val']

        # scannet base init
        self.dataset_config = dataset_config
        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = False
        self.split = split_set

        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.multiview_data = {}

        # MMScan QA task init

        self.task_name = 'embodied_qa'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len

        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab,
                                                       add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'

        ## super configuration
        self.tokenizer_config = dict(max_length=self.max_des_len,
                                     padding='max_length',
                                     truncation='longest_first',
                                     return_tensors='np')

        # downsample for quick evaluation
        self.MMScan_loader = MMScan(version='v1', split=split_set, verbose=True,\
            task='MMScan-QA', ratio = 0.1 if split_set=='val' else 1.0 )

        # only need this for convenient evaluation
        self.eval_func = evaluate
        self.annotations = self.MMScan_loader.samples

    def __len__(self):

        return len(self.MMScan_loader)

    def __getitem__(self, idx):

        data_sample_dict = self.parse_dict(self.MMScan_loader[idx])

        return data_sample_dict

    def parse_dict(self, data_dict) -> dict:

        idx = data_dict['index']
        task_name = self.task_name
        question = data_dict['question'].lower()

        input_bboxes = []
        if data_dict['input_bboxes'] is not None:
            assert len(data_dict['input_bboxes']) == len(
                data_dict['input_bboxes_id'])
            for bbox in data_dict['input_bboxes']:
                input_bboxes.append(torch.tensor(bbox))
            # make it coordinated
            input_bboxes = input_bboxes[:self.max_prompts]
            ret_dict = self._get_scan_data(data_dict['ori_pcds'],
                                           data_dict['bboxes'],
                                           input_bbox=input_bboxes)
            boxes = self._encode_box_coords(ret_dict)
        else:
            ret_dict = self._get_scan_data(data_dict['ori_pcds'],
                                           data_dict['bboxes'],
                                           input_bbox=None)
            boxes = None

        task_key = 'with_box' if boxes is not None else 'without_box'
        prompt = deepcopy(TASK_PROPMT[task_name][task_key])
        prompt['instruction'] = prompt['instruction'].format(locations=boxes,
                                                             question=question)

        # if data_dict["input_bboxes"] is not None:
        #     print(len(data_dict["input_bboxes"]))
        #     print(prompt['instruction'])

        if self.split == 'train':
            caption = data_dict['answers'][0]
            response = prompt['answer'].format(answer=caption)
        else:
            caption = ''
            response = ''

        prompt_inputs = self.tokenizer.batch_encode_plus(
            [prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus(
            [prompt['instruction']], **self.tokenizer_config)

        ## input_ids as labels for LLM
        llm_inputs = self.tokenizer.batch_encode_plus([
            ' '.join(
                (prompt['instruction'], response, self.tokenizer.eos_token))
        ], **self.tokenizer_config)

        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts, ))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts, ))

        if data_dict['input_bboxes'] is not None:
            if random.random() > 0.5:
                # use box to identify an object
                for _index in range(len(input_bboxes)):
                    box_query[_index] = ret_dict['input_box_corners'][
                        _index].reshape(8, 3).astype(np.float32)
                    box_mask[_index] = 1
            else:
                # use click to identify an object
                for _index in range(len(input_bboxes)):
                    click_query[_index] = ret_dict['input_box_centers'][
                        _index].reshape(3, ).astype(np.float32)
                    click_mask[_index] = 1
        else:
            box_query = np.zeros((self.max_prompts, 8, 3))
            box_mask = np.zeros((self.max_prompts, ))
            click_query = np.zeros((self.max_prompts, 3))
            click_mask = np.zeros((self.max_prompts, ))

        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)

        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(
            np.float32)
        ret_dict['gradient_mask'] = (
            llm_inputs['attention_mask'][0] -
            prompt_inputs['attention_mask'][0]).astype(np.float32)

        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(
            np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][
            0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(
            np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][
            0].astype(np.float32)

        keys_to_remove = [k for k in ret_dict.keys() if 'input_box' in str(k)]
        for k in keys_to_remove:
            ret_dict.pop(k)
        return ret_dict

    def _encode_box_coords(self, ret_dict):

        # TODO: output the pcd and the box info here to check if they are match, ensure that's correct

        center_normalized = ret_dict['input_box_centers_normalized']
        size_normalized = ret_dict['input_box_sizes_normalized']
        box_normalized = np.hstack(
            (center_normalized, size_normalized))  # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)

    def _get_scan_data(self, ori_pcds, data_bboxes, input_bbox=None):

        MAX_NUM_OBJ = self.dataset_config.max_num_obj

        # points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        pcd_data = ori_pcds

        mesh_vertices = np.concatenate((pcd_data[0], pcd_data[1]), axis=1)
        instance_labels = pcd_data[-1]

        # semantic is no use
        semantic_labels = pcd_data[-1]

        #try:
        instance_bboxes = [
            np.array(data_bboxes[obj_id]['bbox']) for obj_id in data_bboxes
        ]
        instance_bboxes = [
            instance_bbox[:6] for instance_bbox in instance_bboxes
        ]
        instance_bboxes = np.stack(instance_bboxes)
        # except:
        #     print([np.array(data_bboxes[obj_id]["bbox"]).shape for obj_id in data_bboxes])
        #     print([instance_bbox.shape for instance_bbox in instance_bboxes])
        #     import json
        #     with open("write_log.json","w") as f:
        #         json.dump(str([np.array(data_bboxes[obj_id]["bbox"]).shape for obj_id in data_bboxes])+str([instance_bbox.shape for instance_bbox in instance_bboxes]),f)

        if instance_bboxes.shape[0] > MAX_NUM_OBJ:
            instance_bboxes = instance_bboxes[:MAX_NUM_OBJ, :]
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            # skip the div process
            point_cloud[:, 3:] = (point_cloud[:, 3:] * 256.0 -
                                  MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_normal:
            normals = np.zeros(
                mesh_vertices[:, 0:3].shape)  #mesh_vertices[:,6:9].shape
            point_cloud = np.concatenate([point_cloud, normals], 1)
        assert point_cloud.size > 0

        # adding a height-feature
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ, ), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ, ), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ, ), dtype=np.float32)
        object_ids = np.zeros((MAX_NUM_OBJ, ), dtype=np.int64)

        point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                       self.num_points,
                                                       return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        if input_bbox is not None:

            input_bbox = np.array([np.array(t) for t in input_bbox])
            input_bbox = np.array([_bbox[:6] for _bbox in input_bbox])
            if len(input_bbox.shape) == 1:
                input_bbox = np.expand_dims(input_bbox, axis=0)

        # augment: rotation and filp
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                if input_bbox is not None:
                    input_bbox[:, 0] = -1 * input_bbox[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                if input_bbox is not None:
                    input_bbox[:, 1] = -1 * input_bbox[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi /
                         18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3],
                                         np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat)
            if input_bbox is not None:
                input_bbox = self.dataset_config.rotate_aligned_boxes(
                    input_bbox, rot_mat)

        raw_sizes = target_bboxes[:, 3:6]

        if input_bbox is not None:
            embodied_scan_raw_sizes = input_bbox[:, 3:6]

        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[
            ..., None]

        if input_bbox is not None:
            embodied_scan_box_centers = input_bbox.astype(np.float32)[:, 0:3]
            embodied_scan_box_centers_normalized = shift_scale_points(
                embodied_scan_box_centers[None, ...],
                src_range=[
                    point_cloud_dims_min[None, ...],
                    point_cloud_dims_max[None, ...],
                ],
                dst_range=self.center_normalizing_range,
            )
            embodied_scan_box_centers_normalized = embodied_scan_box_centers_normalized.squeeze(
                0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)
        if input_bbox is not None:
            embodied_scan_box_sizes_normalized = scale_points(
                embodied_scan_raw_sizes.astype(np.float32)[None, ...],
                mult_factor=1.0 / mult_factor[None, ...],
            )
            embodied_scan_box_sizes_normalized = embodied_scan_box_sizes_normalized.squeeze(
                0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)
        object_ids[:instance_bboxes.shape[0]] = instance_bboxes[:, -1]

        if input_bbox is not None:
            embodied_scan_box_corners = self.dataset_config.box_parametrization_to_corners_np(
                embodied_scan_box_centers[None, ...],
                embodied_scan_raw_sizes.astype(np.float32)[None, ...],
                np.zeros((embodied_scan_box_centers.shape[0], ),
                         dtype=np.float32)[None, ...],
            )
            embodied_scan_box_corners = embodied_scan_box_corners.squeeze(0)

        ret_dict = {}
        if input_bbox is not None:
            ret_dict['input_box_corners'] = embodied_scan_box_corners.astype(
                np.float32)
            ret_dict['input_box_centers'] = embodied_scan_box_centers.astype(
                np.float32)
            ret_dict[
                'input_box_centers_normalized'] = embodied_scan_box_centers_normalized.astype(
                    np.float32)
            ret_dict[
                'input_box_sizes_normalized'] = embodied_scan_box_sizes_normalized.astype(
                    np.float32)
        else:
            ret_dict['input_box_corners'] = np.zeros(
                (MAX_NUM_OBJ, 24)).astype(np.float32)
            ret_dict['input_box_centers'] = np.zeros(
                (MAX_NUM_OBJ, 3)).astype(np.float32)
            ret_dict['input_box_centers_normalized'] = np.zeros(
                (MAX_NUM_OBJ, 3)).astype(np.float32)
            ret_dict['input_box_sizes_normalized'] = np.zeros(
                (MAX_NUM_OBJ, 3)).astype(np.float32)

        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['gt_box_corners'] = box_corners.astype(np.float32)
        ret_dict['gt_box_centers'] = box_centers.astype(np.float32)
        ret_dict['gt_box_centers_normalized'] = box_centers_normalized.astype(
            np.float32)
        ret_dict['gt_angle_class_label'] = angle_classes.astype(np.int64)
        ret_dict['gt_angle_residual_label'] = angle_residuals.astype(
            np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        ret_dict['gt_box_sem_cls_label'] = target_bboxes_semcls.astype(
            np.int64)
        ret_dict['gt_box_present'] = target_bboxes_mask.astype(np.float32)
        ret_dict['pcl_color'] = pcl_color
        ret_dict['gt_box_sizes'] = raw_sizes.astype(np.float32)
        ret_dict['gt_box_sizes_normalized'] = box_sizes_normalized.astype(
            np.float32)
        ret_dict['gt_box_angles'] = raw_angles.astype(np.float32)
        ret_dict['point_cloud_dims_min'] = point_cloud_dims_min.astype(
            np.float32)
        ret_dict['point_cloud_dims_max'] = point_cloud_dims_max.astype(
            np.float32)
        ret_dict['gt_object_ids'] = object_ids.astype(np.int64)

        return ret_dict


if __name__ == '__main__':
    test = Dataset('', DatasetConfig())
    print(len(test))
