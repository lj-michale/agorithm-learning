# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         TextClassifyTool
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
import re
import csv
import tensorflow as tf
import numpy as np
from gensim.models import word2vec

# 文本清理函数
def text_clearTitle_word2vec(text, n=12):
    text = text.lower()  # 将文本转化成小写
    text = re.sub(r"[^a-z]", " ", text)  # 替换非标准字符，^是求反操作。
    text = re.sub(r" +", " ", text)  # 替换多重空格
    # text = re.sub(" ", "", text) #替换隔断空格
    text = text.strip()  # 取出首尾空格
    text = text + " eos"  # 添加结束符
    text = text.split(" ")
    return text


# 将标签转为one-hot格式函数
def get_label_one_hot(list):
    values = np.array(list)
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


# 获取训练集和标签函数
def get_word2vec_dataset(n=12):
    agnews_label = []
    agnews_title = []
    agnews_train = csv.reader(open("E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\dataset\\train.csv", "r"))
    for line in agnews_train:
        agnews_label.append(np.int(line[0]))
        agnews_title.append(text_clearTitle_word2vec(line[1]))
        # 设置训练参数
    model = word2vec.Word2Vec(agnews_title, size=64, min_count=0, window=5)
    train_dataset = []
    for line in agnews_title:
        length = len(line)
        if length > n:
            line = line[:n]
            word2vec_matrix = (model[line])
            train_dataset.append(word2vec_matrix)
        else:
            word2vec_matrix = (model[line])
            pad_length = n - length
            pad_matrix = np.zeros([pad_length, 64]) + 1e-10
            word2vec_matrix = np.concatenate([word2vec_matrix, pad_matrix], axis=0)
            train_dataset.append(word2vec_matrix)
    train_dataset = np.expand_dims(train_dataset, 3)
    label_dataset = get_label_one_hot(agnews_label)
    return train_dataset, label_dataset


# word2vec_CNN的模型
def word2vec_CNN():
    xs = tf.keras.Input([None, None])
    # 设置卷积核大小为[3,64]通道为12的卷积计算
    conv_3 = tf.keras.layers.Conv2D(12, [3, 64], activation=tf.nn.relu)(xs)
    # 设置卷积核大小为[3,64]通道为12的卷积计算
    conv_5 = tf.keras.layers.Conv2D(12, [5, 64], activation=tf.nn.relu)(conv_3)
    # 设置卷积核大小为[3,64]通道为12的卷积计算
    conv_7 = tf.keras.layers.Conv2D(12, [7, 64], activation=tf.nn.relu)(conv_5)
    # 下面是分别对卷积计算的结果进行池化处理，将池化处理的结果转成二维结构
    conv_3_mean = tf.keras.layers.Flatten(tf.reduce_max(conv_3, axis=1, keep_dims=True))
    conv_5_mean = tf.keras.layers.Flatten(tf.reduce_max(conv_5, axis=1, keep_dims=True))
    conv_7_mean = tf.keras.layers.Flatten(tf.reduce_max(conv_7, axis=1, keep_dims=True))
    flatten = tf.concat([conv_3_mean, conv_5_mean, conv_7_mean], axis=1)  # 连接多个卷积值
    fc_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten)  # 采用全连接层进行分类
    logits = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(fc_1)  # 获取分类数据
    model = tf.keras.Model(inputs=xs, outputs=logits)
    return model


if __name__ == '__main__':
    get_word2vec_dataset(12)