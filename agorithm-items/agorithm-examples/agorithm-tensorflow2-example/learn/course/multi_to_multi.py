# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         multi_to_multi
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)

iris_data_1 = []
iris_data_2 = []

for iris in iris_data:
    iris_data_1.append(iris[:2])
    iris_data_2.append(iris[2:])

iris_label = data.target
iris_target = np.float32(tf.keras.utils.to_categorical(data.target, num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices(((iris_data_1, iris_data_2), iris_target)).batch(128)

input_xs_1 = tf.keras.Input(shape=2, name="input_xs_1")  # 接收输入参数一
input_xs_2 = tf.keras.Input(shape=2, name="input_xs_2")  # 接收输入参数二
input_xs = tf.concat([input_xs_1, input_xs_2], axis=-1)
out = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(input_xs)
out = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(out)

logist = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(out)
label = tf.keras.layers.Dense(1, name="label")(out)

model = tf.keras.Model(inputs=[input_xs_1, input_xs_2], outputs=[logist, label])
opt = tf.optimizers.Adam(1e-3)


def my_MSE(y_true, y_pred):
    """
    MSE
    :param y_true:
    :param y_pred:
    :return:
    """
    my_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return my_loss


model.compile(optimizer=tf.optimizers.Adam(1e-3),
              loss={"predictions": tf.losses.categorical_crossentropy, "label": my_MSE},
              loss_weights={"predictions": 0.1, "label": 0.5},
              metrics=["accuracy"])

model.fit(x=train_data, epochs=500)

score = model.evaluate(train_data)
print("多头score：", score)

