# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         iris_analysis_with_keras
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf

data = load_iris()

iris_target = data.target
iris_data = np.float32(data.data)

print(iris_data)
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#     ......
#  [6.3 2.5 5.  1.9]
#  [5.9 3.  5.1 1.8]]

# one-hot处理
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))
iris_data = tf.data.Dataset.from_tensor_slices(iris_data).batch(50)
iris_target = tf.data.Dataset.from_tensor_slices(iris_target).batch(50)

model = tf.keras.models.Sequential()

# add layers
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="softmax"))

opt = tf.optimizers.Adam(1e-3)

for epoch in range(1000):
    for _data, label in zip(iris_data, iris_target):
        with tf.GradientTape() as tapes:
            logits = model(_data)
            loss_value = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=label, y_pred=logits))
            grads = tapes.gradient(loss_value, model.trainable_variables)
            print("training loss is :", loss_value.numpy())



