# Copyright (c) OpenRobotLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union, Any
from terminaltables import AsciiTable
import logging
import numpy as np
import torch
from lib.euler_utils import euler_iou3d_split
from tqdm import tqdm

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap

def abbr(sub_class):
    sub_class = sub_class.lower()
    sub_class = sub_class.replace('single', 'sngl')
    sub_class = sub_class.replace('inter', 'int')
    sub_class = sub_class.replace('unique', 'uniq')
    sub_class = sub_class.replace('common', 'cmn')
    sub_class = sub_class.replace('attribute', 'attr')
    if 'sngl' in sub_class and ('attr' in sub_class or 'eq' in sub_class):
        sub_class = 'vg_sngl_attr'
    return sub_class


def ground_eval_subset(gt_anno_list, det_anno_list, logger=None, prefix=''):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
    assert len(det_anno_list) == len(gt_anno_list)
    iou_thr = [0.25, 0.5]
    num_samples = len(gt_anno_list) # each sample contains multiple pred boxes
    # these lists records for each sample, whether a gt box is matched or not
    gt_matched_records = [[] for _ in iou_thr]
    # these lists records for each pred box, NOT for each sample        
    sample_indices = [] # each pred box belongs to which sample
    confidences = [] # each pred box has a confidence score
    ious = [] # each pred box has a ious, shape (num_gt) in the corresponding sample
    # record the indices of each reference type
    num_gts_per_sample = []

    for sample_idx in tqdm(range(num_samples)):
        det_anno = det_anno_list[sample_idx]
        gt_anno = gt_anno_list[sample_idx]

        target_scores = det_anno['score']  # (num_query, )
        top_idxs =  torch.argsort(target_scores, descending=True)
        target_scores = target_scores[top_idxs]
        pred_center = det_anno['center'][top_idxs]
        pred_size = det_anno['size'][top_idxs]
        pred_rot = det_anno['rot'][top_idxs]
        
        gt_center = gt_anno['center']
        gt_size = gt_anno['size']
        gt_rot = gt_anno['rot']

        num_preds = pred_center.shape[0]
        num_gts = gt_center.shape[0]
        num_gts_per_sample.append(num_gts)
        iou_mat = euler_iou3d_split(pred_center, pred_size, pred_rot, gt_center, gt_size, gt_rot)
        for i, score in enumerate(target_scores):
            sample_indices.append(sample_idx)
            confidences.append(score)
            ious.append(iou_mat[i])

    subset_result = {
        'confidences': confidences, # list, num_preds
        'sample_indices': sample_indices, # list, num_preds
        'ious': ious, # list, num_preds, each element is a num_gt (changing) np.ndarray
        'num_gts_per_sample': num_gts_per_sample, # list, batch_size
        'prefix': prefix
    }
    return subset_result

def ground_eval_overall(gt_anno_list, det_anno_list, logger=None):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
    iou_thr = [0.25, 0.5]
    reference_options = list(set(abbr(gt_anno['sub_class']) for gt_anno in gt_anno_list))
    reference_options.sort()
    assert len(det_anno_list) == len(gt_anno_list)
    full_result = {}
    for ref in reference_options:
        indices = [i for i, gt_anno in enumerate(gt_anno_list) if abbr(gt_anno['sub_class']) == ref]
        sub_gt_annos = [gt_anno_list[i] for i in indices ]
        sub_det_annos = [det_anno_list[i] for i in indices ]
        subset_results = ground_eval_subset(sub_gt_annos, sub_det_annos, logger=logger, prefix=ref)
        full_result[ref] = subset_results
    # overall results
    # contatenate all the subsets' results
    full_result['overall'] = {
        'confidences': [],
        'sample_indices': [],
        'ious': [],
        'num_gts_per_sample': [],
        'prefix': 'overall'
    }
    prev_samples = 0
    for ref in reference_options:
        full_result['overall']['confidences'].extend(full_result[ref]['confidences'])
        full_result['overall']['sample_indices'].extend([x + prev_samples for x in full_result[ref]['sample_indices']])
        full_result['overall']['ious'].extend(full_result[ref]['ious'])
        full_result['overall']['num_gts_per_sample'].extend(full_result[ref]['num_gts_per_sample'])
        prev_samples += len(full_result[ref]['num_gts_per_sample'])
    return full_result

def compute_metric_subset(subset_results, iou_thr=[0.25, 0.5], logger=None, prefix=''):
    assert isinstance(subset_results, dict), f"expected a dict, but got {type(subset_results)}"
    confidences = subset_results['confidences']
    sample_indices = subset_results['sample_indices']
    ious = subset_results['ious']
    total_gt_boxes =  sum(subset_results['num_gts_per_sample'])
    gt_matched_records = [[] for thr in iou_thr]
    for iou_idx in range(len(iou_thr)):
        for n_gt in subset_results['num_gts_per_sample']:
            gt_matched_records[iou_idx].append(np.zeros(n_gt, dtype=bool))

    confidences = np.array(confidences)
    sorted_inds = np.argsort(-confidences)
    sample_indices = [sample_indices[i] for i in sorted_inds]
    ious = [ious[i] for i in sorted_inds]

    tp_thr = {}
    fp_thr = {}
    for thr in iou_thr:
        tp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))
        fp_thr[f'{prefix}@{thr}'] = np.zeros(len(sample_indices))

    for d, sample_idx in enumerate(sample_indices):
        iou_max = -np.inf
        cur_iou = ious[d]
        num_gts = cur_iou.shape[0]
        if num_gts > 0:
            for j in range(num_gts):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thr in enumerate(iou_thr):
            if iou_max >= thr:
                if not gt_matched_records[iou_idx][sample_idx][jmax]:
                    gt_matched_records[iou_idx][sample_idx][jmax] = True
                    tp_thr[f'{prefix}@{thr}'][d] = 1.0
                else:
                    fp_thr[f'{prefix}@{thr}'][d] = 1.0
            else:
                fp_thr[f'{prefix}@{thr}'][d] = 1.0

    ret = {}
    for t in iou_thr:
        metric = prefix + '@' + str(t)
        fp = np.cumsum(fp_thr[metric])
        tp = np.cumsum(tp_thr[metric])
        recall = tp / float(total_gt_boxes)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret[metric] = float(ap)
        best_recall = recall[-1] if len(recall) > 0 else 0
        f1s = 2 * recall * precision / np.maximum(recall + precision, np.finfo(np.float64).eps)
        best_f1 = max(f1s)
        ret[metric + '_rec'] = float(best_recall)
        ret[metric + '_f1'] = float(best_f1)
    ret[prefix + '_num_gt'] = total_gt_boxes
    return ret

def ground_eval(gt_anno_list, det_anno_list, logger=None):
    """
        det_anno_list: list of dictionaries with keys:
            'bboxes_3d': (N, 9) or a (list, tuple) (center, size, rotmat): (N, 3), (N, 3), (N, 3, 3)
            'target_scores_3d': (N, )
        gt_anno_list: list of dictionaries with keys:
            'gt_bboxes_3d': (M, 9) or a (list, tuple) (center, size, rotmat): (M, 3), (M, 3), (M, 3, 3)
            'sub_class': str
    """
   
    
    iou_thr = [0.25, 0.5]
    reference_options = [abbr(gt_anno.get('sub_class', 'other')) for gt_anno in gt_anno_list]
    reference_options = list(set(reference_options))
    reference_options.sort()
    reference_options.append('overall')
    
    
    assert len(det_anno_list) == len(gt_anno_list)
    metric_results = {}
    mid_result = ground_eval_overall(gt_anno_list, det_anno_list, logger=logger)
    for ref in reference_options:
        metric_results.update(compute_metric_subset(
            mid_result[ref], iou_thr=iou_thr, logger=logger, prefix=ref))
    metric_results.update(compute_metric_subset(
        mid_result['overall'], iou_thr=iou_thr, logger=logger, prefix='overall'))
    
    header = ['Type']
    header.extend(reference_options)
    table_columns = [[] for _ in range(len(header))]
    for t in iou_thr:
        table_columns[0].append('AP  '+str(t))
        table_columns[0].append('Rec '+str(t))            
        table_columns[0].append('F1 '+str(t))            
        for i, ref in enumerate(reference_options):
            metric = ref + '@' + str(t)
            ap = metric_results[metric]
            best_recall = metric_results[metric + '_rec']
            best_f1 = metric_results[metric + '_f1']
            table_columns[i+1].append(f'{float(ap):.4f}')
            table_columns[i+1].append(f'{float(best_recall):.4f}')
            table_columns[i+1].append(f'{float(best_f1):.4f}')
    table_columns[0].append('Num GT')            
    for i, ref in enumerate(reference_options):
        # add num_gt
        table_columns[i+1].append(f'{int(metric_results[ref + "_num_gt"])}')

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table_data = [list(row) for row in zip(*table_data)] # transpose the table
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    # print('\n' + table.table)
    if logger is not None:
        logger.write('\n' + table.table + '\n')
        logger.flush()
    print('\n' + table.table)

    return metric_results