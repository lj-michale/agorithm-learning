# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Word2VecCNN
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf


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
