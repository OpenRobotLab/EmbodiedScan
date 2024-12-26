import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from easydict import EasyDict as edict
from model.pointnext.cpp import pointnet2_cuda
from omegaconf import OmegaConf
from torch.autograd import Function

logger = get_logger(__name__)

CHANNEL_MAP = {
    'fj': lambda x: x,
    'df': lambda x: x,
    'assa': lambda x: x * 3,
    'assa_dp': lambda x: x * 3 + 3,
    'dp_fj': lambda x: 3 + x,
    'pj': lambda x: x,
    'dp': lambda x: 3,
    'pi_dp': lambda x: x + 3,
    'pj_dp': lambda x: x + 3,
    'dp_fj_df': lambda x: x * 2 + 3,
    'dp_fi_df': lambda x: x * 2 + 3,
    'pi_dp_fj_df': lambda x: x * 2 + 6,
    'pj_dp_fj_df': lambda x: x * 2 + 6,
    'pj_dp_df': lambda x: x + 6,
    'dp_df': lambda x: x + 3,
}

# activation

_ACT_LAYER = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    leakyrelu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
)


def create_act(act_args):
    """Build activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if act_args is None:
        return None
    act_args = copy.deepcopy(act_args)

    if isinstance(act_args, str):
        act_args = {'act': act_args}

    act = act_args.pop('act', None)
    if act is None:
        return None

    if isinstance(act, str):
        act = act.lower()
        assert act in _ACT_LAYER.keys(), f'input {act} is not supported'
        act_layer = _ACT_LAYER[act]

    inplace = act_args.pop('inplace', True)

    if act not in ['gelu', 'sigmoid']:  # TODO: add others
        return act_layer(inplace=inplace, **act_args)
    else:
        return act_layer(**act_args)


# norm


class LayerNorm1d(nn.LayerNorm):
    """LayerNorm for channels of '1D' spatial BCN tensors."""

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2,
                                      1), self.normalized_shape, self.weight,
                            self.bias, self.eps).permute(0, 2, 1).contiguous()


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial BCHW tensors."""

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape,
                            self.weight, self.bias,
                            self.eps).permute(0, 3, 2, 1).contiguous()


class FastBatchNorm1d(nn.Module):
    """Fast BachNorm1d for input with shape [B, N, C], where the feature
    dimension is at last.

    Borrowed from torch-points3d: https://github.com/torch- points3d/torch-
    points3d
    """

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def _forward_dense(self, x):
        return self.bn(x.transpose(1, 2)).transpose(2, 1)

    def _forward_sparse(self, x):
        return self.bn(x)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError('Non supported number of dimensions {}'.format(
                x.dim()))


_NORM_LAYER = dict(
    bn1d=nn.BatchNorm1d,
    bn2d=nn.BatchNorm2d,
    bn=nn.BatchNorm2d,
    in2d=nn.InstanceNorm2d,
    in1d=nn.InstanceNorm1d,
    gn=nn.GroupNorm,
    syncbn=nn.SyncBatchNorm,
    ln=nn.LayerNorm,  # for tokens
    ln1d=LayerNorm1d,  # for point cloud
    ln2d=LayerNorm2d,  # for point cloud
    fastbn1d=FastBatchNorm1d,
    fastbn2d=FastBatchNorm1d,
    fastbn=FastBatchNorm1d,
)


def create_norm(norm_args, channels, dimension=None):
    """Build normalization layer.

    Returns:
        nn.Module: Created normalization layer.
    """
    if norm_args is None:
        return None
    if isinstance(norm_args, dict):
        norm_args = edict(copy.deepcopy(norm_args))
        norm = norm_args.pop('norm', None)
    else:
        norm = norm_args
        norm_args = edict()
    if norm is None:
        return None
    if isinstance(norm, str):
        norm = norm.lower()
        if dimension is not None:
            dimension = str(dimension).lower()
            if dimension not in norm:
                norm += dimension
        assert norm in _NORM_LAYER.keys(), f'input {norm} is not supported'
        norm = _NORM_LAYER[norm]
    return norm(channels, **norm_args)


# conv


class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv1d, self).__init__(*args, 1, **kwargs)
        else:
            super(Conv1d, self).__init__(*args, **kwargs)


class Conv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv2d, self).__init__(*args, (1, 1), **kwargs)
        else:
            super(Conv2d, self).__init__(*args, **kwargs)


def create_convblock1d(*args,
                       norm_args=None,
                       act_args=None,
                       order='conv-norm-act',
                       **kwargs):
    out_channels = args[1]
    in_channels = args[0]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv1d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f'{order} is not supported')

    return nn.Sequential(*conv_layer)


def create_convblock2d(*args,
                       norm_args=None,
                       act_args=None,
                       order='conv-norm-act',
                       **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv2d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f'{order} is not supported')

    return nn.Sequential(*conv_layer)


# group


class KNN(nn.Module):

    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B,
                                        C,
                                        nfeatures,
                                        nsample,
                                        device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample,
                                            features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N],
                                    dtype=torch.float,
                                    device=grad_out.device,
                                    requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample,
                                                 grad_out_data, idx,
                                                 grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample,
                                   device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample,
                                          new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):

    def __init__(self,
                 radius: float,
                 nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2  # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self,
                query_xyz: torch.Tensor,
                support_xyz: torch.Tensor,
                features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans,
                                         idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(
                -1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(
            features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self,
                new_xyz: torch.Tensor,
                xyz: torch.Tensor,
                features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(
            2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):

    def __init__(self,
                 nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self,
                query_xyz: torch.Tensor,
                support_xyz: torch.Tensor,
                features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans,
                                         idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(
                -1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(
                torch.sum(grouped_xyz**2, dim=1)),
                                      dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def create_grouper(group_args):
    # group_args_copy = copy.deepcopy(group_args)
    group_args_copy = copy.deepcopy(OmegaConf.to_object(group_args))
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logger.debug(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample, **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([
            p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]),
            dp, fj, df
        ], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


# subsample


def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Uses iterative furthest point sampling to select a set of npoint
        features that have the largest minimum distance.

        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp,
                                                       output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply

# upsample


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the three nearest neighbors of unknown in known.

        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """Performs weight linear interpolation on 3 features.

        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2_cuda.three_interpolate_wrapper(B, c, m, n, features, idx,
                                                 weight, output)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.zeros([B, c, m],
                                    device='cuda',
                                    requires_grad=True)
        grad_out_data = grad_out.data.contiguous()

        pointnet2_cuda.three_interpolate_grad_wrapper(B, c, n, m,
                                                      grad_out_data, idx,
                                                      weight,
                                                      grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def three_interpolation(unknown_xyz, known_xyz, know_feat):
    """
    input: known_xyz: (m, 3), unknown_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats
