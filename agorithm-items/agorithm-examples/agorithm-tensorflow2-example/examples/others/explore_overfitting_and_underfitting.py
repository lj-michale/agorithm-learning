# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         explore_overfitting_and_underfitting
# Description:  过拟合与欠拟合
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
# 过拟合了。在训练集上达到较高的准确率是容易的，但我们的目的是在测试集，即模型没有见过的数据集上表现良好。

# 过拟合的反面是欠拟合。即对于测试数据还存在改进的空间。通常由如下原因导致：
# 模型不够好。
# 过度正则化(over-regularized)
# 训练时间过短。
# 这些都意味着模型没有学习到训练数据的特征。

# 如果训练太久，模型可能开始过拟合，导致在测试集上表现不佳，因此我们需要取得一个平衡

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb

import numpy as np
import matplotlib.pyplot as plt

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20

N = 10000


def multi_hot_encoding(sentences, dim=10000):
    results = np.zeros((len(sentences), dim))
    for i, word_indices in enumerate(sentences):
        results[i, word_indices] = 1.0
    return results


(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=N)
train_x = multi_hot_encoding(train_x)
test_x = multi_hot_encoding(test_x)

plt.plot(train_x[0])
plt.show()

# 过拟合


def build_and_train(hidden_dim, regularizer=None, dropout=0):
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim, activation='relu',
                           input_shape=(N,),
                           kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(hidden_dim, activation='relu',
                           kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
    history = model.fit(train_x, train_y, epochs=10, batch_size=512,
                        validation_data=(test_x, test_y), verbose=0)

    return history


baseline_history = build_and_train(16)
smaller_history = build_and_train(4)
larger_history = build_and_train(512)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(10, 6))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name + ' 验证集')
        plt.plot(history.epoch, history.history[key],
                 color=val[0].get_color(), label=name + ' 训练集')

    plt.xlabel('Epochs')
    plt.ylabel('Loss - ' + key)
    plt.legend()

    plt.xlim([0, max(history.epoch)])


plot_history([('基线', baseline_history),
              ('较小', smaller_history),
              ('较大', larger_history)])


# 如何防止过拟合
l2_model_history = build_and_train(16, keras.regularizers.l2(0.001))
dpt_model_history = build_and_train(16, dropout=0.2)
plot_history([('基线', baseline_history),
              ('L2正则', l2_model_history),
              ('Dropout', dpt_model_history)])

