from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# All potential activation func list
# TODO: inplace=True
ACTIVATE = [nn.ReLU(), nn.SiLU(), nn.LeakyReLU()]


def same_shape_pad(k: int, d: int = 1):  # TODO: what about the stride
    """
    Calculates the padding size of a conv layer to maintain the input shape.
    :param k: kernel size
    :param d: dilation size
    :return: padding size
    """
    k = d * (k - 1) + 1 if d > 1 else k
    p = k // 2  # floor div
    return p


class Linear(nn.Module):
    """Linear / Fully-connected layer."""

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=bias)

    def forward(self, x):
        return self.linear(x)


class Conv(nn.Module):
    """2D convolution layer, with batch norm and activation."""

    def __init__(self, c_in, c_out, k=1, s=1, p=None, d=1, bias=False):
        """
        Initializes the 2D convolution layer.
        :param c_in: input channels
        :param c_out: output channels
        :param k: kernel size
        :param p: padding, if p is None, pad to the same size by default
        :param s: stride
        :param d: dilation size
        """
        super().__init__()
        p = same_shape_pad(k, d) if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.activate = ACTIVATE[0]  # TODO: optimize the activation func.

    def forward(self, x):
        """
        [B, C_in, H_in, W_in] -> [B, C_out, H_out, W_out].
        """
        return self.activate(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Bottleneck block in ResNet."""

    def __init__(self, c_in, c_hidden, s, expansion=4):
        """
        Initializes the Bottleneck block.
        :param c_in: input channels
        :param c_hidden: hidden channels
        :param s: stride for the 2nd conv layer (could be s=2 for feature map downsample)
        :param expansion: expansion factor of the output channels to the hidden channels
        """
        super().__init__()
        c_out = c_hidden * expansion
        self.conv_1 = Conv(c_in, c_hidden, k=1, s=1)
        self.conv_2 = Conv(c_hidden, c_hidden, k=3, s=s)
        self.conv_3 = Conv(c_hidden, c_out, k=1, s=1)
        self.bn_hidden = nn.BatchNorm2d(c_hidden)
        self.bn = nn.BatchNorm2d(c_out)
        self.activate = ACTIVATE[0]

        # Shortcut connection / skip connection
        if c_in == c_out and s == 1:  # input and output with same spatial size and feature dim
            self.shortcut = nn.Identity()  # no BN for identity
        else:
            self.shortcut = nn.Sequential(Conv(c_in, c_out, k=1, s=s), self.bn) # TODO: check

    def forward(self, x):
        """
        [B, C_in, H_in, W_in] -> [B, C_out, H_out, W_out].
        """
        x_id = x  # identity of x, for shortcut connection
        x_out = self.activate(self.bn_hidden(self.conv_1(x)))
        x_out = self.activate(self.bn_hidden(self.conv_2(x_out)))
        x_out = self.bn(self.conv_3(x_out))
        x_shortcut = self.shortcut(x_id)
        x_out = self.activate(x_out + x_shortcut)
        return x_out


class ResNetLayer(nn.Module):
    """A standard ResNet layer with n Bottleneck blocks."""

    def __init__(self, c_in, c_hidden, n_blocks, downsample=True, expansion=4):
        """
        Initializes the ResNet layer.
        :param c_in: input channels
        :param c_hidden: hidden channels
        :param n_blocks: number of bottleneck blocks in resnet layer
        :param expansion: expansion factor of the output channels to the hidden channels
        :param downsample: whether to downsample the input feature map in the first block
        """
        super().__init__()
        c_out = c_hidden * expansion

        # Handel the first bottleneck block
        if downsample:
            blocks = [Bottleneck(c_in, c_hidden, s=2, expansion=expansion)]
        else:
            blocks = [Bottleneck(c_in, c_hidden, s=1, expansion=expansion)]

        # Handel the rest (n_blocks - 1)
        blocks.extend([Bottleneck(c_out, c_hidden, s=1, expansion=expansion)] * (n_blocks - 1))
        self.resnet_layer = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resnet_layer(x)


class AvgPool(nn.Module):
    """Average pooling layer, global pooling or given kernel and stride."""

    def __init__(self, glb: bool=True, k=None, s=None, p=None):
        """
        Initialize the average pooling layer.
        :param glb: global pooling or local pooling
        :param k: kernel size (optional, only when glb is False)
        :param s: stride (optional, only when glb is False)
        :param p: padding (optional, only when glb is False, if None, use same shape padding size)
        """
        super().__init__()
        if glb:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            p = same_shape_pad(k) if p is None else p
            self.pool = nn.AvgPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.pool(x)


class MaxPool(nn.Module):
    """Max pooling layer, global pooling or given kernel and stride."""

    def __init__(self, glb: bool=True, k=None, s=None, p=None):
        """
        Initialize the max pooling layer.
        :param glb: global pooling or local pooling
        :param k: kernel size (optional, only when glb is False)
        :param s: stride (optional, only when glb is False)
        :param p: padding (optional, only when glb is False, if None, use same shape padding size)
        """
        super().__init__()
        if glb:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            p = same_shape_pad(k) if p is None else p
            self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    """Up-sampling layer by interpolation."""

    def __init__(self, scale, mode='bilinear'):
        """
        Initializes the up-sampling layer.
        :param scale: up-sampling scale factor
        :param mode: "nearest", "bilinear"
        """
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Flatten(nn.Module):

    def __init__(self, dim_s=1, dim_e=-1):
        super().__init__()
        self.flatten = nn.Flatten(dim_s, dim_e)

    def forward(self, x):
        return self.flatten(x)

class Concat(nn.Module):
    """Concatenates a list of tensors along a given dimension."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.dim)