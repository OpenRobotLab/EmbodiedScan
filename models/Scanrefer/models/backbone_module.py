import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), 'lib'))  # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import (PointnetFPModule,
                                             PointnetSAModuleVotes)


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleVotes(npoint=2048,
                                         radius=0.2,
                                         nsample=64,
                                         mlp=[input_feature_dim, 64, 64, 128],
                                         use_xyz=True,
                                         normalize_xyz=True)

        self.sa2 = PointnetSAModuleVotes(npoint=1024,
                                         radius=0.4,
                                         nsample=32,
                                         mlp=[128, 128, 128, 256],
                                         use_xyz=True,
                                         normalize_xyz=True)

        self.sa3 = PointnetSAModuleVotes(npoint=512,
                                         radius=0.8,
                                         nsample=16,
                                         mlp=[256, 128, 128, 256],
                                         use_xyz=True,
                                         normalize_xyz=True)

        self.sa4 = PointnetSAModuleVotes(npoint=256,
                                         radius=1.2,
                                         nsample=16,
                                         mlp=[256, 128, 128, 256],
                                         use_xyz=True,
                                         normalize_xyz=True)

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """

        pointcloud = data_dict['point_clouds']

        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(
            xyz, features)  # this fps_inds is just 0,1,...,1023
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(
            xyz, features)  # this fps_inds is just 0,1,...,511
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(
            xyz, features)  # this fps_inds is just 0,1,...,255
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'],
                            data_dict['sa3_features'],
                            data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'],
                            data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict[
            'sa1_inds'][:, 0:
                        num_seed]  # indices among the entire input point clouds
        return data_dict


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
