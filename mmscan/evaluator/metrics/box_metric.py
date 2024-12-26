from typing import Dict, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def average_precision(recalls: np.ndarray,
                      precisions: np.ndarray,
                      mode: str = 'area') -> np.ndarray:
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
            Defaults to 'area'.

    Returns:
        np.ndarray: Calculated average precision.
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


def get_f1_scores(iou_matrix: Union[np.ndarray, torch.tensor],
                  iou_threshold) -> float:
    """Refer to the algorithm in Multi3DRefer to compute the F1 score.

    Args:
        iou_matrix (ndarray/tensor):
            The iou matrix of the predictions and ground truths with
                shape (num_preds , num_gts)
        iou_threshold (float): 0.25/0.5

    Returns:
        float: the f1 score as the result
    """
    iou_thr_tp = 0
    pred_bboxes_count, gt_bboxes_count = iou_matrix.shape

    square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
    iou_matrix_fill = np.zeros(shape=(square_matrix_len, square_matrix_len),
                               dtype=np.float32)
    iou_matrix_fill[:pred_bboxes_count, :gt_bboxes_count] = iou_matrix

    # apply matching algorithm
    row_idx, col_idx = linear_sum_assignment(iou_matrix_fill * -1)

    # iterate matched pairs, check ious
    for i in range(pred_bboxes_count):
        iou = iou_matrix[row_idx[i], col_idx[i]]
        # calculate true positives
        if iou >= iou_threshold:
            iou_thr_tp += 1

    # calculate precision, recall and f1-score for the current scene
    f1_score = 2 * iou_thr_tp / (pred_bboxes_count + gt_bboxes_count)

    return f1_score


def __get_fp_tp_array__(iou_array: Union[np.ndarray, torch.tensor],
                        iou_threshold: float) \
                        -> Tuple[np.ndarray, np.ndarray]:
    """Compute the False-positive and True-positive array for each prediction.

    Args:
        iou_array (ndarray/tensor):
            the iou matrix of the predictions and ground truths
            (shape num_preds, num_gts)
        iou_threshold (float): 0.25/0.5

    Returns:
        np.ndarray, np.ndarray: (len(preds)),
        the false-positive and true-positive array for each prediction.
    """
    gt_matched_records = np.zeros((len(iou_array[0])), dtype=bool)
    tp_thr = np.zeros((len(iou_array)))
    fp_thr = np.zeros((len(iou_array)))

    for d, _ in enumerate(range(len(iou_array))):
        iou_max = -np.inf
        cur_iou = iou_array[d]
        num_gts = cur_iou.shape[0]

        if num_gts > 0:
            for j in range(num_gts):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        if iou_max >= iou_threshold:
            if not gt_matched_records[jmax]:
                gt_matched_records[jmax] = True
                tp_thr[d] = 1.0
            else:
                fp_thr[d] = 1.0
        else:
            fp_thr[d] = 1.0

    return fp_thr, tp_thr


def subset_get_average_precision(subset_results: dict,
                                 iou_thr: float)\
                                 -> Tuple[np.ndarray, np.ndarray]:
    """Return the average precision and max recall for a given iou array,
    "subset" version while the num_gt of each sample may differ.

    Args:
        subset_results (dict):
            The results, consisting of scores, sample_indices, ious.
            sample_indices means which sample the prediction belongs to.
        iou_threshold (float): 0.25/0.5

    Returns:
        Tuple[np.ndarray, np.ndarray]: the average precision and max recall.
    """
    confidences = subset_results['scores']
    sample_indices = subset_results['sample_indices']
    ious = subset_results['ious']
    gt_matched_records = {}
    total_gt_boxes = 0
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx not in gt_matched_records:
            gt_matched_records[sample_idx] = np.zeros((len(ious[i]), ),
                                                      dtype=bool)
            total_gt_boxes += ious[i].shape[0]

    confidences = np.array(confidences)
    sorted_inds = np.argsort(-confidences)
    sample_indices = [sample_indices[i] for i in sorted_inds]
    ious = [ious[i] for i in sorted_inds]

    tp_thr = np.zeros(len(sample_indices))
    fp_thr = np.zeros(len(sample_indices))

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

        if iou_max >= iou_thr:
            if not gt_matched_records[sample_idx][jmax]:
                gt_matched_records[sample_idx][jmax] = True
                tp_thr[d] = 1.0
            else:
                fp_thr[d] = 1.0
        else:
            fp_thr[d] = 1.0

    fp = np.cumsum(fp_thr)
    tp = np.cumsum(tp_thr)
    recall = tp / float(total_gt_boxes)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return average_precision(recall, precision), np.max(recall)


def get_average_precision(iou_array: np.ndarray, iou_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Return the average precision and max recall for a given iou array.

    Args:
        iou_array (ndarray/tensor):
            The iou matrix of the predictions and ground truths
            (shape len(preds)*len(gts))
        iou_threshold (float): 0.25/0.5

    Returns:
        Tuple[np.ndarray, np.ndarray]: the average precision and max recall.
    """
    fp, tp = __get_fp_tp_array__(iou_array, iou_threshold)
    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recall = tp_cum / float(iou_array.shape[1])
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    return average_precision(recall, precision), np.max(recall)


def get_general_topk_scores(iou_array: Union[np.ndarray, torch.tensor],
                            iou_threshold: float,
                            mode: str = 'sigma') -> Dict[str, float]:
    """Compute the multi-topk metric, we provide two modes.

    Args:
        iou_array (ndarray/tensor):
            the iou matrix of the predictions and ground truths
            (shape len(preds)*len(gts))
        iou_threshold (float): 0.25/0.5
        mode (str): 'sigma'/'simple'
                "simple": 1/N * Hit(min(N*k,len(pred)))
                "sigma": 1/N * Sigma [Hit(min(n*k,len(pred)))>=n] n = 1~N
                    Hit(M) return the number of gtound truths hitted by
                    the first M predictions.
                    N = the number of gtound truths
                Default to 'sigma'.

    Returns:
        Dict[str,float]: the score of multi-topk metric.
    """

    assert mode in ['sigma', 'simple']
    topk_scores = []
    gt_matched_records = np.zeros(len(iou_array[0]))
    num_gt = len(gt_matched_records)
    for d, _ in enumerate(range(len(iou_array))):
        iou_max = -np.inf
        cur_iou = iou_array[d]

        for j in range(len(iou_array[d])):
            iou = cur_iou[j]
            if iou > iou_max:
                iou_max = iou
                j_max = j
        if iou_max >= iou_threshold:
            gt_matched_records[j_max] = True
        topk_scores.append(gt_matched_records.copy())

    topk_results = {}
    for topk in [1, 3, 5, 10]:
        if mode == 'sigma':
            scores = [
                int(
                    np.sum(topk_scores[min(n * topk, len(topk_scores)) -
                                       1]) >= n) for n in range(1, num_gt + 1)
            ]
            result = np.sum(scores) / num_gt
        else:
            query_index = min(num_gt * topk, len(topk_scores)) - 1
            result = np.sum(topk_scores[query_index]) / num_gt
        topk_results[f'gTop-{topk}@{iou_threshold}'] = result
    return topk_results
