# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Helper functions and class to calculate Average Precisions for 3D object
detection."""
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import torch
from utils.box_util import (extract_pc_in_box3d, flip_axis_to_camera_np,
                            get_3d_box, get_3d_box_batch)
from utils.eval_det import eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def get_ap_config_dict(
    remove_empty_box=True,
    use_3d_nms=True,
    nms_iou=0.25,
    use_old_type_nms=False,
    cls_nms=True,
    per_class_proposal=True,
    use_cls_confidence_only=False,
    conf_thresh=0.05,
    no_nms=False,
    dataset_config=None,
):
    """Default mAP evaluation settings for VoteNet."""

    config_dict = {
        'remove_empty_box': remove_empty_box,
        'use_3d_nms': use_3d_nms,
        'nms_iou': nms_iou,
        'use_old_type_nms': use_old_type_nms,
        'cls_nms': cls_nms,
        'per_class_proposal': per_class_proposal,
        'use_cls_confidence_only': use_cls_confidence_only,
        'conf_thresh': conf_thresh,
        'no_nms': no_nms,
        'dataset_config': dataset_config,
    }
    return config_dict


def ez_extract_pc_in_box3d(pc, box3d):
    box_min = box3d.min(0)  # 3
    box_max = box3d.max(0)  # 3
    per_axis_mask = np.hstack(
        (pc > box_min.reshape(1, 3), pc < box_max.reshape(1, 3)))
    per_point_mask = per_axis_mask.sum(-1) == 6
    pc_in_box = pc[per_point_mask]
    inds = np.arange(len(pc))[per_point_mask]
    return pc_in_box, inds


# This is exactly the same as VoteNet so that we can compare evaluations.
def parse_predictions(predicted_boxes,
                      sem_cls_probs,
                      objectness_probs,
                      point_cloud,
                      config_dict=get_ap_config_dict()):
    """Parse predictions to OBB parameters and suppress overlapping boxes.

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    pred_sem_cls = np.argmax(sem_cls_probs, -1)
    obj_prob = objectness_probs.detach().cpu().numpy()

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy()

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                # pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                pc_in_box, inds = ez_extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1
        # -------------------------------------

    if 'no_nms' in config_dict and config_dict['no_nms']:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict['nms_iou'],
                config_dict['use_old_type_nms'],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict['nms_iou'],
                config_dict['use_old_type_nms'],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i,
                    j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict['nms_iou'],
                config_dict['use_old_type_nms'],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    return pred_mask
