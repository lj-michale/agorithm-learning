# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         AGNews
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
import csv
import numpy as np

from gensim.models import Word2Vec
from learn.course.course001.TextClassifyTool import text_clearTitle_word2vec


def get_word2vec_dataset(n=12):
    # 创建标签列表
    agnews_label = []
    # 创建标题列表
    agnews_title = []
    agnews_train = csv.reader(open("E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\dataset\\train.csv", "r"))

    # 将数据读取对应列表中
    for line in agnews_train:
        agnews_label.append(np.int(line[0]))
        # 先将数据进行清洗之后再读取
        agnews_title.append(text_clearTitle_word2vec(line[1]))
    # 设置训练参数
    model = Word2Vec(agnews_title, size=64, min_count=0, window=5)
    # 创建训练集列表
    train_dataset = []

    for line in agnews_title:       # 对长度进行判定
        length = len(line)        # 获取列表长度
        if length > n:            # 对列表长度进行判断
            line = line[:n]       # 截取需要的长度列表
            word2vec_matrix = (model[line])   # 获取word2vec矩阵
            train_dataset.append(word2vec_matrix)    # 将word2vec矩阵添加到训练集中
            # 补全长度不够的操作
        else:
            # 获取word2vec矩阵
            word2vec_matrix = (model[line])
            # 获取需要补全的长度
            pad_length = n - length
            pad_matrix = np.zeros([pad_length, 64]) + 1e-10    # 创建补全矩阵并增加一个小数值
            word2vec_matrix = np.concatenate([word2vec_matrix, pad_matrix], axis=0) # 矩阵补全
            train_dataset.append(word2vec_matrix)          # 将word2vec矩阵添加到训练集中
    train_dataset = np.expand_dims(train_dataset, 3)    # 将三维矩阵进行扩展
    label_dataset = get_label_one_hot(agnews_label)    # 转换成onehot矩阵


    return train_dataset, label_dataset


if __name__ == '__main__':
    get_word2vec_dataset(12)