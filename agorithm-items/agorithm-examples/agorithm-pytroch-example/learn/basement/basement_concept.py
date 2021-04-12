# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         basement_concept
# Description:  Pytorch基本概念
# Author:       orange
# Date:         2021/4/11
# -------------------------------------------------------------------------------

import torch
import numpy as np

# #####################  初始化 Tensor  #####################
torch.tensor([[1., -1.], [1., -1.]])

# NumPy(array) -> Tensor
a = torch.tensor(np.array([[1, -1], [1, -1]]))
print("a.dtype:{}", a.dtype)

# 特殊张量
# torch.zero_([2, 4], dtype=torch.int32)
torch.ones([2, 4])
torch.eye(4)
torch.arange(start=0, end=1, step=0.2)
torch.linspace(start=0, end=1, steps=5)
torch.empty(2, 3)
torch.full(size=[2, 3], fill_value=0.5)
torch.rand(2, 5)
torch.randint(low=0, high=10, size=(3, 4))
torch.randn(3, 4)

b = torch.rand(2, 5)
torch.zeros_like(a)
c = a.zero_()       # 将a矩阵原地清零，并返回0

# ##################### Tensor数学运算  #####################
d = torch.randn(2, 3)
torch.abs(a)
# torch.exp(a)
# torch.sigmoid(a)
torch.clamp(a, min=0, max=1)   # 截断元素值在[0, 1]之间

# ##################### Tensor索引分片合并变换  ###############
e = torch.randn(3, 4)
torch.index_select(e, dim=1, index=torch.tensor([0, 2]))
e[:, [0, 2]]

# ##################### 在GPU上计算  ##########################
torch.cuda.is_available()     # 判断是否可以使用支持CUDA的GPU, CPU张量与GPU张量不能够运算
print(torch.cuda.is_available())
f = torch.randn(2, 3)
# g = f.t('cuda')
# h = f.to('cuda:1')            # 将a张量移动到另一个GPU上，多个GPU编号从0开始递增

# ##################### 自动微分 Autograd  ##########################
from torch.autograd import Function


class ReLU(Function):          # 继承Function类
    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return (output > 0).float() * grad_output        # 计算反传梯度


x = torch.randn(2, 3, requires_grad=True)
y = ReLU.apply(x)
loss = y.sum()
loss.backward()



















































