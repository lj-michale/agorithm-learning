# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         cnn_opreation
# Description:  PyTorch
# Author:       orange
# Date:         2021/4/11
# -------------------------------------------------------------------------------

import torch

a = torch.arange(16).view(4, 4)
print(a)


def conv2d(x, weight, bias, stride, pad):
    """
    :param x:
    :param weight:
    :param bias:
    :param stride:
    :param pad:
    :return:
    """
    n, c, h_in, w_in = x.shape
    d, c, k, j = weight.shape

    x_pad = torch.zeros(n, c, h_in+2*pad, w_in+2*pad).to(x.device)
    x_pad[:, :, pad:-pad, pad:-pad]
    x_pad = x_pad.unfold(2, k, stride)
    x_pad = x_pad.unfold(3, j, stride)
    out = torch.einsum(
        'nchwkj, dckj->ndhw',
        x_pad, weight
        )
    out = out + bias.view(1, -1, 1, 1)
    return out


x = torch.randn(2, 3, 5, 5, requires_grad=True)
w = torch.randn(4, 3, 3, 3, requires_grad=True)
b = torch.randn(4, requires_grad=True)
stride = 2
pad = 2
torch_out = conv2d(x, w, b, stride, pad)
print(torch_out)