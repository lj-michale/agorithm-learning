# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         WordCNN
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf
from sklearn.model_selection import train_test_split

# 获取数据集
from learn.course.course001.AGNews import get_word2vec_dataset
from learn.course.course001.Word2VecCNN import word2vec_CNN

train_dataset, label_dataset = get_word2vec_dataset()
# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.1, random_state=217)
batch_size = 12
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
# 使用模型进行计算
# model = word2vec_CNN()
# model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
# model.fit(train_data, epochs=1)
# score = model.evaluate(X_test, y_test)
# print("last score:", score)
