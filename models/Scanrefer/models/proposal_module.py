"""Modified from: https://github.com/facebookresearch/votenet/blob/master/model
s/proposal_module.py."""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), 'lib'))  # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes


def normalize_vector(vector):
    norm = torch.norm(vector, dim=1, keepdim=True) + 1e-8
    normalized_vector = vector / norm
    return normalized_vector


def cross_product(a, b):
    cross_product = torch.cross(a, b, dim=1)
    return cross_product


def ortho_6d_2_Mat(x_raw, y_raw):
    """x_raw, y_raw: both tensors batch*3."""
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)  # batch*3
    x = cross_product(y, z)  # batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


class ProposalModule(nn.Module):

    def __init__(self,
                 num_class,
                 num_heading_bin,
                 num_size_cluster,
                 mean_size_arr,
                 num_proposal,
                 sampling,
                 seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.angle_bin = 6
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True)

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.proposal = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(
                128,
                2 + 3 + self.angle_bin + num_size_cluster * 4 + self.num_class,
                1))

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        # Farthest point sampling (FPS) on votes
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)

        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(
            0, 2, 1).contiguous()  # (batch_size, num_proposal, 128)
        data_dict[
            'aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class,
                                       self.num_size_cluster,
                                       self.mean_size_arr)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_size_cluster,
                      mean_size_arr):
        """decode the predicted parameters for the bounding boxes."""
        # here we get: 1. angle info 2. size ratio info

        net_transposed = net.transpose(
            2, 1).contiguous()  # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:, :, 0:2]

        base_xyz = data_dict[
            'aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:, :, 2:
                                           5]  # (batch_size, num_proposal, 3)

        # angle
        x_raw = net_transposed[:, :, 5:5 + 3]
        y_raw = net_transposed[:, :, 5 + 3:5 + 3 + 3]
        batch_size = x_raw.shape[0]
        x_raw = x_raw.view(-1, 3)
        y_raw = y_raw.view(-1, 3)
        rot_mat = ortho_6d_2_Mat(x_raw, y_raw).reshape(batch_size, -1, 3, 3)

        size_scores = net_transposed[:, :, 5 + self.angle_bin:5 +
                                     self.angle_bin + num_size_cluster]
        size_residuals_normalized = net_transposed[:, :, 5 + self.angle_bin +
                                                   num_size_cluster:5 +
                                                   self.angle_bin +
                                                   num_size_cluster * 4].view(
                                                       [
                                                           batch_size,
                                                           num_proposal,
                                                           num_size_cluster, 3
                                                       ]
                                                   )  # Bxnum_proposalxnum_size_clusterx3

        sem_cls_scores = net_transposed[:, :,
                                        5 + self.angle_bin + num_size_cluster *
                                        4:]  # Bxnum_proposalx10

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        # data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
        # data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        # data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin
        data_dict['rot_mat'] = rot_mat
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict[
            'size_residuals'] = size_residuals_normalized * torch.from_numpy(
                mean_size_arr.astype(
                    np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['size_calc'] = (
            size_residuals_normalized + 1) * torch.from_numpy(
                mean_size_arr.astype(
                    np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        return data_dict
