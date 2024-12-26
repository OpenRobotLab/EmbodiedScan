# Copyright (c) Facebook, Inc. and its affiliates.
"""Modified from https://github.com/facebookresearch/votenet Dataset for object
bounding box regression.

An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the
box.
"""
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
BASE = '.'  ## Replace with path to dataset
DATASET_ROOT_DIR = os.path.join(BASE, 'data', 'scannet', 'scannet_data')
DATASET_METADATA_DIR = os.path.join(BASE, 'data', 'scannet', 'meta_data')

# some processes are no-use, just drop it off


class DatasetConfig(object):

    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 128

    def angle2class(self, angle):
        raise ValueError('ScanNet does not have rotated bounding boxes.')

    def class2anglebatch_tensor(self,
                                pred_cls,
                                residual,
                                to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7, ))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size,
                                       box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle,
                                        box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size,
                                          box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class ScanNetBaseDataset(Dataset):

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
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        # assert split_set in ["train", "val"]

        root_dir = DATASET_ROOT_DIR
        meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir

        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        self.multiview_data = {}

    def __len__(self):
        return len(self.scan_names)

    def _get_scan_data(self, scan_name, input_bbox=None):

        MAX_NUM_OBJ = self.dataset_config.max_num_obj

        # points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        pcd_data = self.MMScan_loader.get_possess('point_clouds', scan_name)

        mesh_vertices = np.concatenate((pcd_data[0], pcd_data[1]), axis=1)
        instance_labels = pcd_data[-1]

        # semantic is no use
        semantic_labels = pcd_data[-1]

        # todo: here we should not simple use the [:6], simple transform from euler is OK.

        instance_bboxes = [
            np.array(self.embodied_scan_box_info[scan_name][obj_id]['bbox'])
            for obj_id in self.embodied_scan_box_info[scan_name]
        ]
        instance_bboxes = [
            self.MMScan_loader.down_9DOF_to_6DOF(pcd_data, instance_bbox)
            for instance_bbox in instance_bboxes
        ]
        instance_bboxes = np.stack(instance_bboxes)

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

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(os.path.join(
                    self.data_path, 'enet_feats_maxpool.hdf5'),
                                                     'r',
                                                     libver='latest')
            multiview = self.multiview_data[pid][scan_name]
            point_cloud = np.concatenate([point_cloud, multiview], 1)

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

        ###  skip  ###
        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes,
                [instance_labels, semantic_labels])
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
        ###  skip  ###

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
            input_bbox = np.array([
                self.MMScan_loader.down_9DOF_to_6DOF(pcd_data, _bbox)
                for _bbox in input_bbox
            ])
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
            ret_dict['input_centers_normalized'] = np.zeros(
                (MAX_NUM_OBJ, 3)).astype(np.float32)
            ret_dict['input_sizes_normalized'] = np.zeros(
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

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.

        # point_votes = np.zeros([self.num_points, 3])
        # point_votes_mask = np.zeros(self.num_points)
        # for i_instance in np.unique(instance_labels):
        #     # find all points belong to that instance
        #     ind = np.where(instance_labels == i_instance)[0]
        #     # find the semantic label
        #     if semantic_labels[ind[0]] in self.dataset_config.nyu40ids:
        #         x = point_cloud[ind,:3]
        #         center = 0.5*(x.min(0) + x.max(0))
        #         point_votes[ind, :] = center - x
        #         point_votes_mask[ind] = 1.0
        # point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

        # ret_dict['vote_label'] = point_votes.astype(np.float32)
        # ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)

        return ret_dict

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = self._get_scan_data(scan_name)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        return ret_dict
