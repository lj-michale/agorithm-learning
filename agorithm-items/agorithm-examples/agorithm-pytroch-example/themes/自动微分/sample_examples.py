# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         sample_examples
# Description:
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------

import torch
from torch.autograd import Variable

# 第一类：纯标量
x = Variable(torch.ones(1)*3, requires_grad=True)
y = Variable(torch.ones(1)*4, requires_grad=True)
z = x.pow(2)+3*y.pow(2)    # z = x^2+3y^2, dz/dx=2x, dz/dy=6y
z.backward()               #纯标量结果可不写占位变量
print(x.grad)              # x = 3 时, dz/dx=2x=2*3=6
print(y.grad)              # y = 4 时, dz/dy=6y=6*4=24


