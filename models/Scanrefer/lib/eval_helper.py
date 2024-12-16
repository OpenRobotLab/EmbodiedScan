# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), 'lib'))  # HACK add the lib folder
from copy import deepcopy

from grounding_metric import ground_eval
from lib.ap_helper import parse_predictions
from lib.euler_utils import bbox_to_corners
from lib.loss import SoftmaxRankingLoss
from utils.box_util import box3d_iou, get_3d_box, get_3d_box_batch
from utils.nn_distance import huber_loss, nn_distance

from mmscan import VisualGroundingEvaluator

# here we use the MMScan evaluator for visual grounding tasks.


def eval_ref_one_sample(pred_bbox, gt_bbox):
    """Evaluate one reference prediction.

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [
        sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2
    ]
    y_corners = [
        sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2
    ]
    z_corners = [
        sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2
    ]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def inference(data_dict,
              config,
              use_lang_classifier=False,
              use_oracle=False,
              use_cat_rand=False,
              use_best=False,
              post_processing=None):
    """Loss functions.

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    pred_res = []
    gt_res = []
    batch_size, num_words, _ = data_dict['lang_feat'].shape

    objectness_preds_batch = torch.argmax(data_dict['objectness_scores'],
                                          2).long().detach()

    if post_processing:
        _ = parse_predictions(data_dict, post_processing)
        nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

        # construct valid mask
        pred_masks = (nms_masks * objectness_preds_batch == 1).bool().detach()
    else:
        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).bool().detach()

    pred_center = data_dict['center'].detach().cpu()
    pred_rot_mat = data_dict['rot_mat'].detach().cpu()
    pred_size_class = torch.argmax(data_dict['size_scores'],
                                   -1)  # B,num_proposal
    pred_size = torch.gather(
        data_dict['size_calc'], 2,
        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, 1, 3))  # B,num_proposal,1,3
    pred_size = pred_size.squeeze(2)
    pred_size = pred_size.detach().cpu()
    pred_score = data_dict['cluster_ref'].detach().cpu()
    gt_bbox = data_dict['target_bbox'].cpu()
    gt_rot_mats = data_dict['target_rot_mat'].cpu()
    gt_ref = data_dict['ref_box_label'].cpu()

    for i in range(batch_size):
        mask = pred_masks[i]
        pred_center_single = pred_center[i][mask]
        pred_rot_mat_single = pred_rot_mat[i][mask]
        pred_size_single = pred_size[i][mask]
        pred_score_single = pred_score[i][mask]
        pred_res.append({
            'center': pred_center_single,
            'size': pred_size_single,
            'rot': pred_rot_mat_single,
            'score': pred_score_single
        })

        # TODO check if it only work for single gt box setting
        # rbler: no problem?
        gt_center = gt_bbox[i, gt_ref[i]][:, :3]
        gt_size = gt_bbox[i, gt_ref[i]][:, 3:6]
        gt_rot_mat = gt_rot_mats[i, gt_ref[i]]

        # print("gt_center shape:", gt_center.shape)
        gt_res.append({
            'center': gt_center.cpu(),
            'size': gt_size.cpu(),
            'rot': gt_rot_mat.cpu(),
            'sub_class': data_dict['sub_class'][i],
            'ID': data_dict['sample_ID'][i]
        })

    return pred_res, gt_res


def get_eval(pred_list, gt_list, logger):

    return ground_eval(gt_list, pred_list, logger)
    #return get_eval_new(pred_list,gt_list, logger)


def to_mmscan_form(raw_batch_result):

    batch_result = {}
    batch_result['pred_scores'] = raw_batch_result['pred_list']['score']
    batch_result['pred_bboxes'] = raw_batch_result['pred_list']
    batch_result['gt_bboxes'] = raw_batch_result['gt_list']
    batch_result['index'] = raw_batch_result['index']
    batch_result['subclass'] = raw_batch_result['gt_list']['sub_class']
    batch_result['ID'] = raw_batch_result['gt_list']['ID']
    return batch_result


def get_eval_new(pred_list, gt_list, logger):

    vg_evaluator = VisualGroundingEvaluator(True)

    assert len(pred_list) == len(gt_list)

    batch_input = []
    for _index in range(len(pred_list)):
        batch_input.append({'pred_list':pred_list[_index],\
            'gt_list':gt_list[_index],'index':_index})
        batch_input[-1] = to_mmscan_form(batch_input[-1])

    vg_evaluator.update(batch_input)

    print('Staring evaluation!')
    print(vg_evaluator.start_evaluation())

    result_table = vg_evaluator.print_result()
    vg_evaluator.reset()

    if logger is not None:
        logger.write('\n' + result_table + '\n')
        logger.flush()

    return result_table


if __name__ == '__main__':
    store_file_path = '/mnt/petrelfs/linjingli/tmp/data/big_tmp/result_1020.pt'
    store_results = torch.load(store_file_path)
    pred_list = store_results['pred_list']
    gt_list = store_results['gt_list']
    get_eval(pred_list, gt_list, None)
