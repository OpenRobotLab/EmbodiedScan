from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss

GT_VOTE_FACTOR = 3  # number of GT votes per point


def nn_distance(pc1,
                pc2,
                l1smooth=False,
                delta=1.0,
                l1=False,
                return_distance=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    if return_distance:
        return dist1, idx1, dist2, idx2, pc_dist
    else:
        return dist1, idx1, dist2, idx2


class Matcher(nn.Module):

    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs['sem_cls_prob'].shape[0]
        nqueries = outputs['sem_cls_prob'].shape[1]
        ngt = targets['gt_box_sem_cls_label'].shape[1]
        nactual_gt = targets['nactual_gt']

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs['sem_cls_prob']
        gt_box_sem_cls_labels = (
            targets['gt_box_sem_cls_label'].unsqueeze(1).expand(
                batchsize, nqueries, ngt))
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs['objectness_prob'].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs['center_dist'].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs['gious'].detach()

        final_cost = (self.cost_class * class_mat +
                      self.cost_objectness * objectness_mat +
                      self.cost_center * center_mat +
                      self.cost_giou * giou_mat)

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros([batch_size, nprop],
                                       dtype=torch.int64,
                                       device=pred_cls_prob.device)
        proposal_matched_mask = torch.zeros([batch_size, nprop],
                                            dtype=torch.float32,
                                            device=pred_cls_prob.device)
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(
                    final_cost[b, :, :nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            'assignments': assignments,
            'per_prop_gt_inds': per_prop_gt_inds,
            'proposal_matched_mask': proposal_matched_mask,
        }


class SetPredictionCriterion(nn.Module):

    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super(SetPredictionCriterion, self).__init__()

        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict['loss_no_object_weight']
        del loss_weight_dict['loss_no_object_weight']
        self.register_buffer('semcls_percls_weights', semcls_percls_weights)

        self.loss_functions = {
            'loss_sem_cls': self.loss_sem_cls,
            'loss_angle': self.loss_angle,
            'loss_center': self.loss_center,
            'loss_size': self.loss_size,
            'loss_giou': self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            'loss_cardinality': self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs['sem_cls_logits']
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) !=
                        pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets['nactual_gt'])
        return {'loss_cardinality': card_err}

    def loss_sem_cls(self, outputs, targets, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs['sem_cls_logits']
        gt_box_label = torch.gather(targets['gt_box_sem_cls_label'], 1,
                                    assignments['per_prop_gt_inds'])
        gt_box_label[assignments['proposal_matched_mask'].int() == 0] = (
            pred_logits.shape[-1] - 1)
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction='mean',
        )

        return {'loss_sem_cls': loss}

    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs['angle_logits']
        angle_residual = outputs['angle_residual_normalized']

        if targets['num_boxes_replica'] > 0:
            gt_angle_label = targets['gt_angle_class_label']
            gt_angle_residual = targets['gt_angle_residual_label']
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin)

            # # Non vectorized version
            # assignments = assignments["assignments"]
            # p_angle_logits = []
            # p_angle_resid = []
            # t_angle_labels = []
            # t_angle_resid = []

            # for b in range(angle_logits.shape[0]):
            #     if len(assignments[b]) > 0:
            #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
            #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
            #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
            #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

            # p_angle_logits = torch.cat(p_angle_logits)
            # p_angle_resid = torch.cat(p_angle_resid)
            # t_angle_labels = torch.cat(t_angle_labels)
            # t_angle_resid = torch.cat(t_angle_resid)

            # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
            # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

            gt_angle_label = torch.gather(gt_angle_label, 1,
                                          assignments['per_prop_gt_inds'])
            angle_cls_loss = F.cross_entropy(angle_logits.transpose(2, 1),
                                             gt_angle_label,
                                             reduction='none')
            angle_cls_loss = (angle_cls_loss *
                              assignments['proposal_matched_mask']).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1,
                assignments['per_prop_gt_inds'])
            gt_angle_label_one_hot = torch.zeros_like(angle_residual,
                                                      dtype=torch.float32)
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1)
            angle_reg_loss = huber_loss(angle_residual_for_gt_class -
                                        gt_angle_residual_normalized,
                                        delta=1.0)
            angle_reg_loss = (angle_reg_loss *
                              assignments['proposal_matched_mask']).sum()

            angle_cls_loss /= targets['num_boxes']
            angle_reg_loss /= targets['num_boxes']
        else:
            angle_cls_loss = torch.zeros(1,
                                         device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1,
                                         device=angle_logits.device).squeeze()
        return {
            'loss_angle_cls': angle_cls_loss,
            'loss_angle_reg': angle_reg_loss
        }

    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs['center_dist']
        if targets['num_boxes_replica'] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2,
                assignments['per_prop_gt_inds'].unsqueeze(-1)).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments['proposal_matched_mask']
            center_loss = center_loss.sum()

            if targets['num_boxes'] > 0:
                center_loss /= targets['num_boxes']
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {'loss_center': center_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs['gious']

        # # Non vectorized version
        # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
        # assign = assignments["assignments"]

        # for b in range(gious_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2,
            assignments['per_prop_gt_inds'].unsqueeze(-1)).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments['proposal_matched_mask']
        giou_loss = giou_loss.sum()

        if targets['num_boxes'] > 0:
            giou_loss /= targets['num_boxes']

        return {'loss_giou': giou_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets['gt_box_sizes_normalized']
        pred_box_sizes = outputs['size_normalized']

        if targets['num_boxes_replica'] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(gt_box_sizes[:, :, x], 1,
                                 assignments['per_prop_gt_inds'])
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes,
                                  gt_box_sizes,
                                  reduction='none').sum(dim=-1)

            # zero-out non-matched proposals
            size_loss *= assignments['proposal_matched_mask']
            size_loss = size_loss.sum()

            size_loss /= targets['num_boxes']
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {'loss_size': size_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs['box_corners'],
            targets['gt_box_corners'],
            targets['nactual_gt'],
            rotated_boxes=torch.any(targets['gt_box_angles'] > 0).item(),
            needs_grad=(self.loss_weight_dict['loss_giou_weight'] > 0),
        )

        outputs['gious'] = gious
        center_dist = torch.cdist(outputs['center_normalized'],
                                  targets['gt_box_centers_normalized'],
                                  p=1)
        outputs['center_dist'] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + '_weight'
            if (loss_wt_key in self.loss_weight_dict
                    and self.loss_weight_dict[loss_wt_key] > 0
                ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets,
                                                   assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace('_weight', '')] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace('_weight', '')]
        return assignments, final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets['gt_box_present'].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()),
                                min=1).item()
        targets['nactual_gt'] = nactual_gt
        targets['num_boxes'] = num_boxes
        targets['num_boxes_replica'] = nactual_gt.sum().item(
        )  # number of boxes on this worker for dist training

        assignments, loss, loss_dict = self.single_output_forward(
            outputs['outputs'], targets)

        if 'aux_outputs' in outputs:
            for k in range(len(outputs['aux_outputs'])):
                _, interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs['aux_outputs'][k], targets)

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f'{interm_key}_{k}'] = interm_loss_dict[
                        interm_key]
        return assignments, loss, loss_dict


class VoteQueryCriterion(nn.Module):

    def __init__(self, cfgs):
        super(VoteQueryCriterion, self).__init__()
        self.loss_dict = {
            'loss_vote': (self.loss_vote, 1),
        }

    def loss_vote(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute vote loss: Match predicted votes to GT votes.

        Args:
            end_points: dict (read-only)

        Returns:
            vote_loss: scalar Tensor

        Overall idea:
            If the seed point belongs to an object (votes_label_mask == 1),
            then we require it to vote for the object center.

            Each seed point may vote for multiple translations v1,v2,v3
            A seed point may also be in the boxes of multiple objects:
            o1,o2,o3 with corresponding GT votes c1,c2,c3

            Then the loss for this seed point is:
                min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
        """

        # Load ground truth votes and assign them to seed points
        batch, num_seed, _ = outputs['seed_xyz'].shape
        vote_xyz = outputs['vote_xyz']  # B,num_seed*vote_factor,3
        seed_inds = outputs['seed_inds'].long(
        )  # B,num_seed in [0,num_points-1]

        # Get groundtruth votes for the seed points
        # vote_label_mask: Use gather to select B,num_seed from B,num_point
        #   non-object point has no GT vote mask = 0, object point has mask = 1
        # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
        #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
        seed_gt_votes_mask = torch.gather(targets['vote_label_mask'], 1,
                                          seed_inds)
        seed_inds_expand = seed_inds.view(batch, num_seed,
                                          1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
        seed_gt_votes = torch.gather(targets['vote_label'], 1,
                                     seed_inds_expand)
        seed_gt_votes += outputs['seed_xyz'].repeat(1, 1, 3)

        # Compute the min of min of distance
        vote_xyz_reshape = vote_xyz.view(batch * num_seed, -1, 3)
        seed_gt_votes_reshape = seed_gt_votes.view(batch * num_seed,
                                                   GT_VOTE_FACTOR, 3)
        # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
        dist1, _, dist2, _ = nn_distance(vote_xyz_reshape,
                                         seed_gt_votes_reshape,
                                         l1=True)
        votes_dist, _ = torch.min(dist2, dim=1)
        votes_dist = votes_dist.view(batch, num_seed)

        loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (
            torch.sum(seed_gt_votes_mask.float()) + 1e-6)

        return {'loss_vote': loss}

    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        # assignments = self.compute_label_assignment(outputs, targets)

        loss = torch.zeros(1)[0].to(targets['point_clouds'].device)
        loss_dict = {}
        loss_intermidiate = {}
        for loss_name, (loss_fn, loss_weight) in self.loss_dict.items():

            # loss_intermidiate = loss_fn(outputs, targets, assignments)
            loss_intermidiate = loss_fn(outputs, targets)
            loss_dict.update(loss_intermidiate)

            loss += loss_weight * loss_intermidiate[loss_name]

        loss *= 10

        return loss, loss_intermidiate


class OverallCriterion(nn.Module):

    def __init__(self, args, dataset_config):
        super(OverallCriterion, self).__init__()
        matcher = Matcher(
            cost_class=args.matcher_cls_cost,
            cost_giou=args.matcher_giou_cost,
            cost_center=args.matcher_center_cost,
            cost_objectness=args.matcher_objectness_cost,
        )
        loss_weight_dict = {
            'loss_giou_weight': args.loss_giou_weight,
            'loss_sem_cls_weight': args.loss_sem_cls_weight,
            'loss_no_object_weight': args.loss_no_object_weight,
            'loss_angle_cls_weight': args.loss_angle_cls_weight,
            'loss_angle_reg_weight': args.loss_angle_reg_weight,
            'loss_center_weight': args.loss_center_weight,
            'loss_size_weight': args.loss_size_weight,
        }
        self.set_prediction_loss = SetPredictionCriterion(
            matcher, dataset_config, loss_weight_dict)
        self.vote_query_loss = VoteQueryCriterion(args)

    def forward(self, votenet_outputs: Dict, decoder_outputs: Dict,
                targets: Dict) -> Dict:
        loss_dict = {}
        votenet_loss, votenet_loss_dict = self.vote_query_loss(
            votenet_outputs, targets)
        assignments, set_loss, set_loss_dict = self.set_prediction_loss(
            decoder_outputs, targets)

        loss_dict.update(votenet_loss_dict)
        loss_dict.update(set_loss_dict)

        return assignments, votenet_loss + set_loss, loss_dict


def build_criterion(args, dataset_config):
    return OverallCriterion(args, dataset_config)
