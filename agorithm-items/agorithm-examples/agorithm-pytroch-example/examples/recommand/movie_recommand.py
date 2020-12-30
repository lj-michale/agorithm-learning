# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         movie_recommand
# Description:  https://blog.csdn.net/fjssharpsword/article/details/95583601
# Author:       orange
# Date:         2020/12/30
# -------------------------------------------------------------------------------
'''
@author:
@data: 2019.07.11
@function: Implementing NCF with Torch
           Dataset: Movielen Dataset(ml-1m)
           Evaluating: hitradio,ndcg
           https://arxiv.org/pdf/1708.05031.pdf
           https://github.com/hexiangnan/neural_collaborative_filtering
'''
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import os


class NCFData(torch.utils.data.Dataset):  # define the dataset
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        # Note that the labels are only useful when training, we thus add them in the ng_sample() function.
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        '''
        if self.is_training:
            self.ng_sample()
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels
        '''
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


# define the NCF model, integrating the GMF and MLP
class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(GMF, self).__init__()

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1)


# define the MLP model
class MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(MLP, self).__init__()

        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        return prediction.view(-1)


# define the NCF model, integrating the GMF and MLP
class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        """
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(factor_num * 2, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):

        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


# loading dataset function
def load_dataset(test_num=100):
    train_data = pd.read_csv("/data/fjsdata/ctKngBase/ml/ml-1m.train.rating", \
                             sep='\t', header=None, names=['user', 'item'], \
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open("/data/fjsdata/ctKngBase/ml/ml-1m.test.negative", 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])  # one postive item
            for i in arr[1:]:
                test_data.append([u, int(i)])  # 99 negative items
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


# evaluate function
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


# Setting GPU Enviroment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # using gpu
cudnn.benchmark = True
# construct the train and test datasets
train_data, test_data, user_num, item_num, train_mat = load_dataset()
train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default
test_dataset = NCFData(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
# every user have 99 negative items and one positive items，so batch_size=100
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)
# training and evaluationg
# Setting GPU Enviroment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # using gpu
cudnn.benchmark = True
# training and evaluationg
print("%3s%20s%20s%20s" % ('K', 'Iterations', 'HitRatio', 'NDCG'))
for K in [8, 16, 32, 64]:  # latent factors
    # model = GMF(int(user_num), int(item_num), factor_num=16)
    # model = MLP(int(user_num), int(item_num), factor_num=16, num_layers=3, dropout=0.0)
    model = NCF(int(user_num), int(item_num), factor_num=16, num_layers=3, dropout=0.0)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()
    best_hr, best_ndcg = 0.0, 0.0
    for epoch in range(20):
        model.train()
        train_loader.dataset.ng_sample()
        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG = metrics(model, test_loader, top_k=10)
        # print("HR: {:.3f}\tNDCG: {:.3f}".format(HR, NDCG))
        if HR > best_hr: best_hr = HR
        if NDCG > best_ndcg: best_ndcg = NDCG
    print("%3d%20d%20.6f%20.6f" % (K, 20, best_hr, best_ndcg))