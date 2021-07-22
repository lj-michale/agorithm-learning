# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         course_0002
# Description:  tensorboard --logdir="E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\course000\\logs"
# Author:       orange
# Date:         2021/7/22
# -------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import os


def load_mnist(path):
    """加载本地下载好的mnist数据集"""
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist("E:\\company\\myself\\datasets\\mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0  # 将样本从整数转换为浮点数

# 利用tf.keras.Sequential容器封装网络层，前一层网络的输出默认作为下一层的输入
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),  # 创建一层网络，设置输出节点数为128，激活函数类型为Relu
    tf.keras.layers.Dropout(0.2),  # 在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')])  # Dense层就是所谓的全连接神经网络层

model.summary()

# 为训练选择优化器和损失函数：
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = os.path.join("logs")
# print(log_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
# 定义TensorBoard对象.histogram_freq 如果设置为0，则不会计算直方图。
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# TensorBoard对象作为回调传给model.fit方法
model.fit(x_train, y_train, epochs=8, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

model.save_weights('weight/my_weights', save_format='tf')  # 保存模型