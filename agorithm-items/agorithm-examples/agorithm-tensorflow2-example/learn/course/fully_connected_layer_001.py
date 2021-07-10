# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         fully_connected_layer_001
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

iris_target = (data.target)
iris_target = np.float32(tf.keras.utils.to_categorical(data.target, num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices(((iris_data, iris_target), iris_target)).batch(128)


# 自定义全连接层
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.weight = tf.Variable(tf.random.normal([input_shape[-1], self.output_dim]), name="dense_weight")
        self.bias = tf.Variable(tf.random.normal([self.output_dim]), name="bias_weight")
        super(MyLayer, self).build(input_shape)

    def call(self, input_tensor):
        out = tf.matmul(input_tensor, self.weight) + self.bias
        out = tf.nn.relu(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        return out


input_xs = tf.keras.Input(shape=4, name="input_xs")
out = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(input_xs)
out = MyLayer(32)(out)        # 自定义全连接层
out = MyLayer(48)(out)        # 自定义全连接层
out = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(out)
logits = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(out)

model = tf.keras.Model(inputs=input_xs, outputs=logits)
opt = tf.optimizers.Adam(1e-3)
model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics=["accuracy"])
model.fit(train_data, epochs=1000)
score = model.evaluate(iris_data, iris_target)

print("iris score:", score)

