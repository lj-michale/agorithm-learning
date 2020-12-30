# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         image_classification
# Description:
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------

import torch
import torch.nn
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optimizer

'''
The compose function allows for multiple transforms.
transform.ToTensor() converts our PILImage to a tensor of 
shape (C x H x W) in the range [0, 1]
transform.Normalize(mean, std) normalizes a tensor to a (mean, std) 
for (R, G, B)
'''
_task = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 注意：此处数据集在本地，因此download=False;若需要下载的改为True
# 同样的，第一个参数为数据存放路径
data_path = '../CIFAR_10_zhuanzhi/cifar10'
cifar = CIFAR10(data_path, train=True, download=False, transform=_task)

# 这里只是为了构造取样的角标，可根据自己的思路进行拓展
# 此处使用了前百分之八十作为训练集，百分之八十到九十的作为验证集，后百分之十为测试集
samples_count = len(cifar)
split_train = int(0.8 * samples_count)
split_valid = int(0.9 * samples_count)

index_list = list(range(samples_count))
train_idx, valid_idx, test_idx = index_list[:split_train], index_list[split_train:split_valid], index_list[split_valid:]

# 定义采样器
# create training and validation, test sampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_samlper = SubsetRandomSampler(test_idx)

# create iterator for train and valid, test dataset
trainloader = DataLoader(cifar, batch_size=256, sampler=train_sampler)
validloader = DataLoader(cifar, batch_size=256, sampler=valid_sampler)
testloader = DataLoader(cifar, batch_size=256, sampler=test_samlper)


# 网络设计
class Net(torch.nn.Module):
    """
    网络设计了三个卷积层，一个池化层，一个全连接层
    """

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, 10)

    # 前向传播
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x


if __name__ == "__main__":

    net = Net()  # 实例化网络
    loss_function = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失

    # 定义优化算法
    optimizer = optimizer.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

    # 迭代次数
    for epoch in range(1, 31):
        train_loss, valid_loss = [], []

        net.train()  # 训练开始
        for data, target in trainloader:
            optimizer.zero_grad()  # 梯度置0
            output = net(data)
            loss = loss_function(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss.append(loss.item())

        net.eval()  # 验证开始
        for data, target in validloader:
            output = net(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

        print("Epoch:{}, Training Loss:{}, Valid Loss:{}".format(epoch, np.mean(train_loss), np.mean(valid_loss)))
    print("======= Training Finished ! =========")

    print("Testing Begining ... ")  # 模型测试
    total = 0
    correct = 0
    for i, data_tuple in enumerate(testloader, 0):
        data, labels = data_tuple
        output = net(data)
        _, preds_tensor = torch.max(output, 1)

        total += labels.size(0)
        correct += np.squeeze((preds_tensor == labels).sum().numpy())
    print("Accuracy : {} %".format(correct / total))