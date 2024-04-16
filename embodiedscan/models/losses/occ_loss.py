"""Adapted from SurroundOcc."""

import torch
import torch.nn.functional as F


def occ_multiscale_supervision(gt_occ,
                               ratio,
                               gt_shape,
                               gt_occupancy_masks=None):
    """Produce multi-scale occupancy supervision from ground truth.

    Args:
        gt_occ (list[Tensor]): Ground truth occupancy.
        ratio (int): Downsample ratio for producing the target occupancy
            supervision.
        gt_shape (list): Target supervision shape, consistent with the
            corresponding occupancy prediction shape.
        gt_occupancy_masks (list[Tensor], optional): Visible occupancy
            mask. Defaults to None.

    Return:
        list[Tensor]: The target-scale occupancy supervision.
    """
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3],
                      gt_shape[4]]).to(gt_occ[0].device).type(torch.long)
    for i in range(gt.shape[0]):
        coords = torch.div(gt_occ[i][:, :3].type(torch.long),
                           ratio,
                           rounding_mode='trunc')
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] = gt_occ[i][:, 3]

        if gt_occupancy_masks is not None:
            gt[i][~gt_occupancy_masks[i]] = 255

    return gt


def geo_scal_loss(pred, ssc_target, semantic=True):
    """Geometric scene-class affinity loss.

    Only consider empty and nonempty probabilities.

    Args:
        pred (Tensor): Prediction maps.
        ssc_target (Tensor): Occupancy target.
        semantic (bool): Whether to consider semantic (use softmax)
            when converting the prediction map to the predicted distributions.
            Defaults to True.

    Returns:
        Tensor: Geometric scene-class affinity loss.
    """
    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-6
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1 - nonempty_target) *
            (empty_probs)).sum() / ((1 - nonempty_target).sum() + eps)
    return (F.binary_cross_entropy(precision, torch.ones_like(precision)) +
            F.binary_cross_entropy(recall, torch.ones_like(recall)) +
            F.binary_cross_entropy(spec, torch.ones_like(spec)))


def sem_scal_loss(pred, ssc_target):
    """Semantic scene-class affinity loss.

    Consider probabilities for different categories.

    Args:
        pred (Tensor): Prediction maps.
        ssc_target (Tensor): Occupancy target.

    Returns:
        Tensor: Semantic scene-class affinity loss.
    """
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision))
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall,
                                                     torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum(
                    (1 - p) *
                    (1 - completion_target)) / (torch.sum(1 -
                                                          completion_target))
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity))
                loss_class += loss_specificity
            loss += loss_class
    if count:
        return loss / count
    else:
        # no ssc_target != 255 and in classes of interest
        return 0 * loss
