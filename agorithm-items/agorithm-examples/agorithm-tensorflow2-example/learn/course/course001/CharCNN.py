# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         CharCNN
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf
from sklearn.model_selection import train_test_split

from learn.course.course001.AGNewsTool import get_dataset


def char_CNN():
    xs = tf.keras.Input([])
    conv_1 = tf.keras.layers.Conv1D(1, 3, activation=tf.nn.relu)(xs)  # 第一层卷积
    conv_1 = tf.keras.layers.BatchNormalization(conv_1)
    conv_2 = tf.keras.layers.Conv1D(1, 5, activation=tf.nn.relu)(conv_1)  # 第一层卷积
    conv_2 = tf.keras.layers.BatchNormalization(conv_2)
    conv_3 = tf.keras.layers.Conv1D(1, 5, activation=tf.nn.relu)(conv_2)  # 第一层卷积
    conv_3 = tf.keras.layers.BatchNormalization(conv_3)
    flatten = tf.keras.layers.Flatten()(conv_3)
    fc_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flatten)  # 全连接网络
    logits = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(fc_1)
    model = tf.keras.Model(inputs=xs, outputs=logits)
    return model


train_dataset, label_dataset = get_dataset()
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.1, random_state=217)
batch_size = 12
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
# 使用模型进行计算
model = char_CNN()
model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(train_data, epochs=1)
score = model.evaluate(X_test, y_test)
print("last score:", score)