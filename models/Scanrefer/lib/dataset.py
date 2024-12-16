'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import h5py
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from tqdm import tqdm

from mmscan import MMScan

sys.path.append(os.path.join(os.getcwd(), 'lib'))  # HACK add the lib folder
from data.scannet.model_util_scannet import (ScannetDatasetConfig,
                                             rotate_aligned_boxes,
                                             rotate_aligned_boxes_along_axis)
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
# SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")

# no multi-view
MULTIVIEW_DATA = ''
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, 'glove.p')
UNK_CLASS_ID = 165  # for "object" in es. The actural value is 166, but es counts from 1.

import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')


def tokenize_text(description):
    try:
        tokens = word_tokenize(description)
    except:
        # download punkt package if not exists
        nltk.download('punkt')
        tokens = word_tokenize(description)
    return tokens


from .euler_utils import euler_to_matrix_np
from .utils_read import (NUM2RAW_3RSCAN, RAW2NUM_3RSCAN, apply_mapping_to_keys,
                         read_es_infos, to_scene_id)


class ScannetReferenceDataset(Dataset):

    def __init__(self,
                 es_info_file,
                 vg_raw_data_file,
                 split='train',
                 num_points=40000,
                 use_height=False,
                 use_color=False,
                 use_normal=False,
                 use_multiview=False,
                 augment=False):
        # load the embeding
        with open(GLOVE_PICKLE, 'rb') as f:
            self.glove_embeding = pickle.load(f)
        from data.scannet.meta_data.es_type_id import es_type_dict
        self.raw2label = {k: v - 1 for k, v in es_type_dict.items()}
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.augment = augment
        self.debug = False

        # set the ratio to 0.2 for training and valing
        self.mmscan_loader = MMScan(version='v1',\
            split=split, verbose=True,task='MMScan-VG',ratio=0.2)

    def __len__(self):
        return len(self.mmscan_loader)

    def __getitem__(self, idx):

        return self.parse_dict(self.mmscan_loader[idx])

    def parse_dict(self, data_dict) -> dict:

        start = time.time()

        # propreccessing for anno
        scene_id = data_dict['scan_id']
        object_id = data_dict['target_id']

        assert len(object_id) > 0

        idx = data_dict['index']

        tokens = tokenize_text(data_dict['text'].lower())

        # get language features
        # tokenize the description

        embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
        for token_id in range(CONF.TRAIN.MAX_DES_LEN):
            if token_id < len(tokens):
                token = tokens[token_id]
                if token in self.glove_embeding:
                    embeddings[token_id] = self.glove_embeding[token]
                else:
                    embeddings[token_id] = self.glove_embeding['unk']

        lang_feat = embeddings
        lang_len = len(tokens)
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN

        # data_dict["pcds"] = self.MMScan_collect["scan"][scan_idx]['pcds']
        # data_dict["obj_pcds"] = self.MMScan_collect["scan"][scan_idx]['obj_pcds']
        # data_dict["scene_center"] = self.MMScan_collect["scan"][scan_idx]['scene_center']
        # data_dict["bboxes"]

        # proprocess for pcds and boxxes
        mesh_vertices = data_dict['pcds']
        assert mesh_vertices.shape[1] == 6
        instance_labels = data_dict['instance_labels']

        instance_bboxes = np.zeros((MAX_NUM_OBJ, 8 + 3))

        for index_, obj_id in enumerate(data_dict['bboxes']):
            if index_ >= MAX_NUM_OBJ:
                break
            bbox = data_dict['bboxes'][obj_id]['bbox']
            obj_type = data_dict['bboxes'][obj_id]['type']
            instance_bboxes[index_, :6] = bbox[:6]
            instance_bboxes[index_, 6:9] = bbox[6:]
            if obj_type in self.raw2label:
                instance_bboxes[index_, 9] = self.raw2label[obj_type]
            else:
                if obj_type == 'steps':
                    instance_bboxes[index_, 9] = self.raw2label['step']
            instance_bboxes[index_, 10] = obj_id

        # use color&normal, no multiview

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = point_cloud[:, 3:6] - MEAN_COLOR_RGB / 256.0
            pcl_color = point_cloud[:, 3:6]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA,
                                                     'r',
                                                     libver='latest')

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1)

        point_cloud, choices = random_sampling(point_cloud,
                                               self.num_points,
                                               return_choices=True)
        instance_labels = instance_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 9))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ, ))
        angle_residuals = np.zeros((MAX_NUM_OBJ, ))
        size_classes = np.zeros((MAX_NUM_OBJ, ))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(
            MAX_NUM_OBJ)  # bbox label for reference target
        ref_class_label = np.zeros(DC.num_size_cluster)

        target_bboxes_rot_mat = None  # If target bboxes rot mat is not None, then the target bbox euler angle is not trusted

        if self.split != 'test':
            num_bbox = instance_bboxes.shape[
                0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:9]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.augment and not self.debug:
                # TODO

                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                    target_bboxes[:, 6] = -target_bboxes[:, 6]
                    target_bboxes[:, 8] = -target_bboxes[:, 8]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                    target_bboxes[:, 6] = -target_bboxes[:, 6]
                    target_bboxes[:, 7] = -target_bboxes[:, 7]

                # Rotation along up-axis/Z-axis
                target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:,
                                                                         6:9])
                rot_angle = (np.random.random() * np.pi /
                             18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3],
                                             np.transpose(rot_mat))
                # target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3],
                                               np.transpose(rot_mat))
                target_bboxes_rot_mat = rot_mat @ target_bboxes_rot_mat

                # Scale
                scale_factor = np.random.uniform(0.9, 1.1)
                target_bboxes[:, 0:6] *= scale_factor
                point_cloud[:, 0:3] *= scale_factor

                # Translation
                trans_factor = np.random.normal(scale=np.array(
                    [.1, .1, .1], dtype=np.float32),
                                                size=3).T
                point_cloud[:, 0:3] += trans_factor
                target_bboxes[:, 0:3] += trans_factor

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            for i_instance in np.unique(instance_labels):
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                if len(ind) < 10:
                    continue

                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes,
                                  (1, 3))  # make 3 votes identical

            class_ind = [
                DC.nyu40id2class[int(x)]
                for x in instance_bboxes[:num_bbox, -2]
            ]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[
                0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]

            # construct the reference target label for each bbox
            if target_bboxes_rot_mat is None:
                target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:,
                                                                         6:9])
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                if gt_id in object_id:
                    ref_box_label[i] = 1
                    ref_class_label[DC.nyu40id2class[int(
                        instance_bboxes[i, -2])]] = 1
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points,
                                    9])  # make 3 votes identical
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [
                DC.nyu40id2class[int(x)]
                for x in instance_bboxes[:, -2][0:num_bbox]
            ]
        except KeyError:
            pass

        if target_bboxes_rot_mat is None:
            target_bboxes_rot_mat = euler_to_matrix_np(target_bboxes[:, 6:9])

        parse_data_dict = {}
        parse_data_dict['point_clouds'] = point_cloud.astype(
            np.float32)  # point cloud data including features
        parse_data_dict['lang_feat'] = lang_feat.astype(
            np.float32)  # language feature vectors
        parse_data_dict['lang_len'] = np.array(lang_len).astype(
            np.int64)  # length of each description
        parse_data_dict['center_label'] = target_bboxes.astype(
            np.float32)[:, 0:3]  # (MAX_NUM_OBJ, 3) for GT box center XYZ
        parse_data_dict['heading_class_label'] = angle_classes.astype(
            np.int64
        )  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        parse_data_dict['heading_residual_label'] = angle_residuals.astype(
            np.float32)  # (MAX_NUM_OBJ,)
        parse_data_dict['target_bbox'] = target_bboxes.astype(np.float32)
        parse_data_dict['target_rot_mat'] = target_bboxes_rot_mat.astype(
            np.float32)
        parse_data_dict['size_class_label'] = size_classes.astype(
            np.int64
        )  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        parse_data_dict['size_residual_label'] = size_residuals.astype(
            np.float32)  # (MAX_NUM_OBJ, 3)
        parse_data_dict['num_bbox'] = np.array(num_bbox).astype(np.int64)
        parse_data_dict['sem_cls_label'] = target_bboxes_semcls.astype(
            np.int64)  # (MAX_NUM_OBJ,) semantic class index
        parse_data_dict['box_label_mask'] = target_bboxes_mask.astype(
            np.float32)  # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        parse_data_dict['vote_label'] = point_votes.astype(np.float32)
        parse_data_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        parse_data_dict['scan_idx'] = np.array(idx).astype(np.int64)
        parse_data_dict['pcl_color'] = pcl_color
        parse_data_dict['ref_box_label'] = ref_box_label.astype(
            bool)  # 0/1 reference labels for each object bbox
        parse_data_dict['ref_class_label'] = ref_class_label.astype(np.float32)
        parse_data_dict['pcl_color'] = pcl_color
        parse_data_dict['sub_class'] = data_dict['sub_class']
        parse_data_dict['sample_ID'] = data_dict['ID']
        parse_data_dict['load_time'] = time.time() - start

        return parse_data_dict

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
