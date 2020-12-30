# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         handwriting_recognition
# Description:
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------

import numpy as np
import torch
from torchvision import transforms

_task = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5], [0.5]
    )
])

from torchvision.datasets import MNIST

# 数据集加载
mnist = MNIST('./data', download=False, train=True, transform=_task)

# 训练集和验证集划分
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# create training and validation split
index_list = list(range(len(mnist)))

split_train = int(0.8 * len(mnist))
split_valid = int(0.9 * len(mnist))

train_idx, valid_idx, test_idx = index_list[:split_train], index_list[split_train:split_valid], index_list[split_valid:]

# create sampler objects using SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# create iterator objects for train and valid dataset
trainloader = DataLoader(mnist, batch_size=256, sampler=train_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=valid_sampler)
test_loader = DataLoader(mnist, batch_size=256, sampler=test_sampler)

# design for net
import torch.nn.functional as F


class NetModel(torch.nn.Module):
    def __init__(self):
        super(NetModel, self).__init__()
        self.hidden = torch.nn.Linear(28 * 28, 300)
        self.output = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    net = NetModel()

    from torch import optim

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

    for epoch in range(1, 12):
        train_loss, valid_loss = [], []
        # net.train()
        for data, target in trainloader:
            optimizer.zero_grad()
            # forward propagation
            output = net(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # net.eval()
        for data, target in validloader:
            output = net(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())
        print("Epoch:", epoch, "Training Loss:", np.mean(train_loss), "Valid Loss:", np.mean(valid_loss))

    print("testing ... ")
    total = 0
    correct = 0
    for i, test_data in enumerate(test_loader, 0):
        data, label = test_data
        output = net(data)
        _, predict = torch.max(output.data, 1)

        total += label.size(0)
        correct += np.squeeze((predict == label).sum().numpy())
    print("Accuracy:", (correct / total) * 100, "%")