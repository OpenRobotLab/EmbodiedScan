# Copyright (c) OpenRobotLab. All rights reserved.
from typing import List, Optional, Tuple

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = None
    pass

from embodiedscan.registry import MODELS


@MODELS.register_module()
class ChannelMapper(BaseModule):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or dict],
            optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 kernel_size: int = 1,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                self._make_conv_block(in_channel, out_channels, kernel_size))

    def _make_conv_block(self, in_channels: int, out_channels: int,
                         kernel_size: int) -> nn.Module:
        """Construct DeConv-Norm-Act-Conv-Norm-Act block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Module: With corresponding layers.
        """
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    dimension=3),
            ME.MinkowskiBatchNorm(out_channels), ME.MinkowskiELU())

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        return outs
