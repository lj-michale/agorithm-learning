# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         pytorch_demo_model
# Description:
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


"""
定义一个类，这个类继承于nn.Module，实现两个方法：初始化函数和正向传播
实例化这个类之后，将参数传入这个类中，进行正向传播
"""
"""
If running on Windows and you get a BrokenPipeError, try setting
the num_worker of torch.utils.data.DataLoader() to 0.
"""

class LeNet(nn.Module):
    def __init__(self):
        # super解决在多重继承中调用父类可能出现的问题
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  # 全连接层输入的是一维向量，第一层节点个数120是根据Pytorch官网demo设定
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10因为使用的是cifar10，分为10类

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input (3,32,32)  output(16, 32-5+1=28, 32-5+1)
        x = self.pool1(x)  # output(16, 28/2=14, 28/2)
        x = F.relu((self.conv2(x)))  # output(32, 14-5+1=10, 14-5+1=10)
        x = self.pool2(x)  # output(32, 10/2=5, 10/2=5)
        x = x.view(-1, 32*5*5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = F.relu(self.fc3(x))  # output(10)
        return x
