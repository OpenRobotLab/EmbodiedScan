import torch.nn as nn
from model.pointnetpp.pointnet2_modules import PointnetSAModule


def break_up_pc(pc):
    """Split the pointcloud into xyz positions and features tensors. This
    method is taken from VoteNet codebase
    (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (pc[..., 3:].transpose(1, 2).contiguous()
                if pc.size(-1) > 3 else None)
    return xyz, features


class PointNetPP(nn.Module):
    """Pointnet++ encoder.

    For the hyper parameters please advise the paper
    (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self,
                 sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')

        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(
                PointnetSAModule(
                    npoint=sa_n_points[i],
                    nsample=sa_n_samples[i],
                    radius=sa_radii[i],
                    mlp=sa_mlps[i],
                    bn=bn,
                    use_xyz=use_xyz,
                ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.out_dim = sa_mlps[-1][-1]
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], self.out_dim)

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)

        return self.fc(features.view(features.size(0), -1))
