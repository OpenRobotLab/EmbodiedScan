from functools import reduce

import torch
import torch._utils
import torch.nn as nn
from torch.autograd import Variable

# compatible with PyTorch 0.4.0
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:

    def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                           requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size,
                                              stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):

    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func, self.forward_prepare(input))


class Padding(nn.Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at
    # index [index] in that dimension. If pad<0, index counts from the left.
    # If pad>0 index counts from the right.
    # When nInputDim is provided, inputs larger than that value will be considered batches
    # where the actual dim to be padded will be dimension dim + 1.
    def __init__(self, dim, pad, value, index, nInputDim):
        super(Padding, self).__init__()
        self.value = value
        # self.index = index
        self.dim = dim
        self.pad = pad
        self.nInputDim = nInputDim
        if index != 0:
            raise NotImplementedError('Padding: index != 0 not implemented')

    def forward(self, input):
        dim = self.dim
        if self.nInputDim != 0:
            dim += input.dim() - self.nInputDim
        pad_size = list(input.size())
        pad_size[dim] = self.pad
        padder = Variable(input.data.new(*pad_size).fill_(self.value))

        if self.pad < 0:
            padded = torch.cat((padder, input), dim)
        else:
            padded = torch.cat((input, padder), dim)
        return padded


class Dropout(nn.Dropout):
    """Cancel out PyTorch rescaling by 1/(1-p)"""

    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """Cancel out PyTorch rescaling by 1/(1-p)"""

    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout2d, self).forward(input)


class StatefulMaxPool2d(nn.MaxPool2d):  # object keeps indices and input sizes

    def __init__(self, *args, **kwargs):
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, x):
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = x.size()
        if return_indices:
            return output, indices
        return output


class StatefulMaxUnpool2d(nn.Module):

    def __init__(self, pooling):
        super(StatefulMaxUnpool2d, self).__init__()
        self.pooling = pooling
        self.unpooling = nn.MaxUnpool2d(pooling.kernel_size, pooling.stride,
                                        pooling.padding)

    def forward(self, x):
        return self.unpooling.forward(x, self.pooling.indices,
                                      self.pooling.input_size)


pooling_0 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
pooling_1 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)
pooling_2 = StatefulMaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False)


def create_enet(num_classes):
    enet = nn.Sequential(  # Sequential,
        LambdaMap(
            lambda x: x,  # ConcatTable,
            nn.Conv2d(3, 13, (3, 3), (2, 2), (1, 1), (1, 1), 1),
            pooling_0,
        ),
        LambdaReduce(lambda x, y: torch.cat((x, y), 1)),
        nn.BatchNorm2d(16, 0.001, 0.1, True),
        nn.PReLU(16),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(16,
                              16, (2, 2), (2, 2), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16,
                              64, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(64, 0.001, 0.1, True),
                    Dropout2d(0.01),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                    pooling_1,
                    Padding(0, 48, 0, 0, 3),
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(64,
                              16, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16,
                              64, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(64, 0.001, 0.1, True),
                    Dropout2d(0.01),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(64,
                              16, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16,
                              64, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(64, 0.001, 0.1, True),
                    Dropout2d(0.01),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(64,
                              16, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16,
                              64, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(64, 0.001, 0.1, True),
                    Dropout2d(0.01),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(64,
                              16, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(16, 0.001, 0.1, True),
                    nn.PReLU(16),
                    nn.Conv2d(16,
                              64, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(64, 0.001, 0.1, True),
                    Dropout2d(0.01),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(64),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(64,
                              32, (2, 2), (2, 2), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                    pooling_2,
                    Padding(0, 64, 0, 0, 3),
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              32, (1, 5), (1, 1), (0, 2), (1, 1),
                              1,
                              bias=False),
                    nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              32, (1, 5), (1, 1), (0, 2), (1, 1),
                              1,
                              bias=False),
                    nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (2, 2), (2, 2), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              32, (1, 5), (1, 1), (0, 2), (1, 1),
                              1,
                              bias=False),
                    nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (4, 4), (4, 4), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (8, 8), (8, 8), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              32, (1, 5), (1, 1), (0, 2), (1, 1),
                              1,
                              bias=False),
                    nn.Conv2d(32, 32, (5, 1), (1, 1), (2, 0), (1, 1), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        nn.Sequential(  # Sequential,
            LambdaMap(
                lambda x: x,  # ConcatTable,
                nn.Sequential(  # Sequential,
                    nn.Conv2d(128,
                              32, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32, 32, (3, 3), (1, 1), (16, 16), (16, 16), 1),
                    nn.BatchNorm2d(32, 0.001, 0.1, True),
                    nn.PReLU(32),
                    nn.Conv2d(32,
                              128, (1, 1), (1, 1), (0, 0), (1, 1),
                              1,
                              bias=False),
                    nn.BatchNorm2d(128, 0.001, 0.1, True),
                    Dropout2d(0.1),
                ),
                nn.Sequential(  # Sequential,
                    Lambda(lambda x: x),  # Identity,
                ),
            ),
            LambdaReduce(lambda x, y: x + y),  # CAddTable,
            nn.PReLU(128),
        ),
        # ENCODER END (add classifier)
        nn.Sequential(
            nn.Conv2d(128,
                      num_classes, (1, 1), (1, 1), (0, 0), (1, 1),
                      1,
                      bias=False))
        #nn.Sequential( # Sequential,
        #    LambdaMap(lambda x: x, # ConcatTable,
        #        nn.Sequential( # Sequential,
        #            nn.Conv2d(128, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.ConvTranspose2d(16, 16, (3, 3), (2, 2), (1, 1), (1, 1)),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(64, 0.001, 0.1, True),
        #        ),
        #        nn.Sequential( # Sequential,
        #            Lambda(lambda x: x), # Identity,
        #            nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(64, 0.001, 0.1, True),
        #            StatefulMaxUnpool2d(pooling_2), #SpatialMaxUnpooling,
        #        ),
        #    ),
        #    LambdaReduce(lambda x,y: x+y), # CAddTable,
        #    nn.PReLU(64),
        #),
        #nn.Sequential( # Sequential,
        #    LambdaMap(lambda x: x, # ConcatTable,
        #        nn.Sequential( # Sequential,
        #            nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(64, 0.001, 0.1, True),
        #        ),
        #        nn.Sequential( # Sequential,
        #            Lambda(lambda x: x), # Identity,
        #        ),
        #    ),
        #    LambdaReduce(lambda x,y: x+y), # CAddTable,
        #    nn.PReLU(64),
        #),
        #nn.Sequential( # Sequential,
        #    LambdaMap(lambda x: x, # ConcatTable,
        #        nn.Sequential( # Sequential,
        #            nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            nn.PReLU(16),
        #            nn.Conv2d(16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(64, 0.001, 0.1, True),
        #        ),
        #        nn.Sequential( # Sequential,
        #            Lambda(lambda x: x), # Identity,
        #        ),
        #    ),
        #    LambdaReduce(lambda x,y: x+y), # CAddTable,
        #    nn.PReLU(64),
        #),
        #nn.Sequential( # Sequential,
        #    LambdaMap(lambda x: x, # ConcatTable,
        #        nn.Sequential( # Sequential,
        #            nn.Conv2d(64, 4, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(4, 0.001, 0.1, True),
        #            nn.PReLU(4),
        #            nn.ConvTranspose2d(4, 4, (3, 3), (2, 2), (1, 1), (1, 1)),
        #            nn.BatchNorm2d(4, 0.001, 0.1, True),
        #            nn.PReLU(4),
        #            nn.Conv2d(4, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #        ),
        #        nn.Sequential( # Sequential,
        #            Lambda(lambda x: x), # Identity,
        #            nn.Conv2d(64, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #            StatefulMaxUnpool2d(pooling_1), #SpatialMaxUnpooling,
        #        ),
        #    ),
        #    LambdaReduce(lambda x,y: x+y), # CAddTable,
        #    nn.PReLU(16),
        #),
        #nn.Sequential( # Sequential,
        #    LambdaMap(lambda x: x, # ConcatTable,
        #        nn.Sequential( # Sequential,
        #            nn.Conv2d(16, 4, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(4, 0.001, 0.1, True),
        #            nn.PReLU(4),
        #            nn.Conv2d(4, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1),
        #            nn.BatchNorm2d(4, 0.001, 0.1, True),
        #            nn.PReLU(4),
        #            nn.Conv2d(4, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False),
        #            nn.BatchNorm2d(16, 0.001, 0.1, True),
        #        ),
        #        nn.Sequential( # Sequential,
        #            Lambda(lambda x: x), # Identity,
        #        ),
        #    ),
        #    LambdaReduce(lambda x,y: x+y), # CAddTable,
        #    nn.PReLU(16),
        #),
        #nn.ConvTranspose2d(16, num_classes, (2, 2), (2, 2), (0, 0), (0, 0)),
    )
    return enet


def create_enet_for_3d(num_2d_classes, model_path, num_3d_classes):
    model = create_enet(num_2d_classes)
    model.load_state_dict(torch.load(model_path))
    # remove the classifier
    n = len(model)
    model_trainable = nn.Sequential(*(model[i] for i in range(n - 9, n - 1)))
    model_fixed = nn.Sequential(*(model[i] for i in range(n - 9)))
    #model_classifier = nn.Sequential(nn.Conv2d(128, num_3d_classes, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias=False))
    model_classifier = nn.Sequential(model[n - 1])
    #print 'model_fixed'
    #print model_fixed
    #print 'model_trainable'
    #print model_trainable
    #print 'model_classifier'
    #print model_classifier
    #raw_input('sdflkj')
    for param in model_fixed.parameters():
        param.requires_grad = False
    return model_fixed, model_trainable, model_classifier
