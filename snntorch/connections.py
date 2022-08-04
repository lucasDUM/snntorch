import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module

import math
import warnings

from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

__all__ = [
    'Identity',
    'Linear_Burst',
    'Linear_Phase',
    'Conv2d_Burst'
]

def repeat_to_length(phase, wanted):
    return (np.tile(phase, (wanted//len(phase) + 1)))[:wanted]


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear_Burst(Module):
    """Applies a linear transformation with threshold adaption to the incoming data: :math:`y = V_th . xA^T + b`
    """
    __constants__ = ['in_features', 'out_features', 'burst_constant']
    in_features: int
    out_features: int
    burst_constant: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, burst_constant: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_Burst, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.burst_constant = burst_constant
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.prev_spike = torch.tensor(0)
        self.First = True

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def burst_function(self, burst_constant, input_):
        self.prev_spike = input_
        if self.First:
            self.First = False
            burst_modifier = torch.ones_like(input_)
        else:
            # Check for each neuron if it fired previously
            mask = torch.eq(input_, self.prev_spike, out=None)
            # Add burst scaling
            modifier = burst_constant*mask
            # Turn any 0s back into 1s since this will be element wise multiplied later
            burst_modifier = modifier[modifier==0] = 1
        # Adapted threshold
        return burst_modifier
    
    def forward(self, step: int, input: Tensor) -> Tensor:
        print(step)
        if step==0:
            self.First=True
        print(self.First)
        # Add burst re_weighting here
        # Input data will be in shape Batch, channel, image
        # The Input data will be a spike train
        # Return Threshold as well
        threshold = self.burst_function(self.burst_constant, input)
        return F.linear(input*threshold, self.weight, self.bias), threshold

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Linear_Phase(Module):
    """Applies a linear transformation with threshold adaption to the incoming data: :math:`y = V_th . xA^T + b`
    """
    __constants__ = ['in_features', 'out_features', 'burst_constant']
    in_features: int
    out_features: int
    burst_constant: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, burst_constant: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_Burst, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.burst_constant = burst_constant
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.First = True
        self.counter = 0

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def create_signal(size, pattern, premade, amplitude=1, offset=0):
        if pattern:
            if len(pattern) > size:
                pattern = pattern[:, size]
        else:
            # Digital signals
            if premade == "simple1":  
                pattern = np.array([1, 0.5, 0, 0.5, 1])
            elif premade == "simple2":
                pattern = np.array([1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1])
            elif premade == "complex1":
                pattern = np.array([2, 1.5, 1, 0.5, 0, 0.5, 1, 1.5, 2])
            elif premade == "complex2":
                pattern = np.array([2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
            else:
                pattern = np.array([1, 0.5, 0, 0.5, 1])

        pattern = pattern*amplitude
        pattern = np.roll(pattern, offset)
        phase = torch.tensor(repeat_to_length(pattern, size))

        return phase

    def phase_function(self, period, step, input):
        if self.First:
            #break
            pass
        return 0

    def forward(self, input: Tensor) -> Tensor:
        # Add burst re_weighting here
        # Input data will be in shape Batch, channel, image
        # The Input data will be a spike train
        return F.linear(input * self.burst_function(self.burst_constant, input), self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    burst_constant: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 burst_constant: int,
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.burst_constant = burst_constant
        self.prev_spike = torch.tensor(0)
        self.First = True

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_Burst(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        burst_constant: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d_Burst, self).__init__(
            in_channels, out_channels, kernel_size_, burst_constant, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def burst_function(self, burst_constant, input_):
        self.prev_spike = input_
        if self.First:
            self.First = False
            burst_modifier = torch.ones_like(input_)
        else:
            mask = torch.eq(input_, self.prev_spike, out=None)
            modifier = burst_constant*mask
            burst_modifier = modifier[modifier==0] = 1
        return burst_modifier

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        threshold = self.burst_function(self.burst_constant, input)
        return self._conv_forward(input*threshold, self.weight, self.bias), threshold