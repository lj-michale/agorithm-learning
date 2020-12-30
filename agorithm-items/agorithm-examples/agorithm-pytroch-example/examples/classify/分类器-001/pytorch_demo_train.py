# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         pytorch_demo_train
# Description:
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------

import torch
import torchvision
import torch.nn as nn
from .pytorch_demo_model import LeNet
import matplotlib as plt
import torchvision.transforms as transforms


import numpy as np

batch_size = 36
learning_rate = 1e-3

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 标准化 output = (input- 0.5)/0.5
)

# 50000张训练图片
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)  # 当前目录的data文件夹下

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 在windows下，num_workers只能设置为0

# 10000张测试图片
testset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)  # 当前目录的data文件夹下

testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True, num_workers=0)  # 在windows下，num_workers只能设置为0

test_data_iter = iter(testloader)  # 将testloader转换为迭代器
test_img, test_label = test_data_iter.next()  # 通过next（）获得一批数据

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def imshow(img):
    img = img / 2 + 0.5  # unnormalize反标准化过程input = output*0.5 + 0.5
    npimg = img.numpy()  # 转换为numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Pytorch内Tensor顺序[batch, channel, height, width],由于输入没有batch，故channel对于0，height对应1，width对应2
    # 此处要还原为载入图像时基础的shape，所以应把顺序变为[height, width, channel]， 所以需要np.transpose(npimg, (1, 2, 0))
    plt.show()


# 打印几张图片看看
# print labels
# print(''.join('%5s' % classes[test_label[j]] for j in range(4))) 此处应将testloader内的batch_size改为4即可，没必要显示10000张
# show images
# imshow(torchvision.utils.make_grid(test_img))

# 实例化
Mynet = LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Mynet.parameters(), lr=learning_rate)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.
    for step, data in enumerate(trainloader, start=0):  # enumerate返回每一批数据和对应的index
        # get the inputs: data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = Mynet(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():  # with是一个上下文管理器
                outputs = Mynet(test_img)  # [batch, 10]
                y_pred = torch.max(outputs, dim=1)[1]  # 找到最大值，即最有可能的类别，第0个维度对应batch，所以dim=1，第一个维度对应类别，[1]代表只需要index即可，即位置
                accuracy = (y_pred == test_label).sum().item() / test_label.size(0)  # 整个预测是在tensor变量中计算的，所以要用.item()转为数值, test_label.size(0)为测试样本的数目

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))  # 500次的平均train_loss
                running_loss = 0.  # 清零，进行下一个500次的计算

print("Training finished")
save_path = './Lenet.pth'
torch.save(Mynet.state_dict(), save_path)
