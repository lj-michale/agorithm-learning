# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         multi_to_one
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
    iris_data_1.append(iris[0])
    iris_data_2.append(iris[1:4])

iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))

train_data = tf.data.Dataset.from_tensor_slices(((iris_data_1, iris_data_2), iris_target)).batch(128)

input_xs_1 = tf.keras.Input(shape=(1,), name="input_xs_1")  # 接收输入参数一
input_xs_2 = tf.keras.Input(shape=(3,), name="input_xs_2")  # 接收输入参数二
input_xs = tf.concat([input_xs_1, input_xs_2], axis=-1)
out = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(input_xs)
out = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(out)

logist = tf.keras.layers.Dense(3, activation="relu", name="predictions")(out)
model = tf.keras.Model(inputs=[input_xs_1, input_xs_2], outputs=logist)
model.compile(optimizer=tf.optimizers.Adam(1e-3),
              loss=tf.losses.categorical_crossentropy, metrics=["accuracy"])
model.fit(x=train_data, epochs=500)
score = model.evaluate(train_data)
print("多头score:", score)




