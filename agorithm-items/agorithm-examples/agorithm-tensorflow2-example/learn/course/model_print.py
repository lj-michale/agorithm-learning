# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         model_print
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
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices(((iris_data, iris_target), iris_target)).batch(128)


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        """
        :param output_dim:
        """
        self.output_dim = output_dim
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        """
        build
        :param input_shape:
        :return:
        """
        self.weight = tf.Variable(tf.random.normal([input_shape[-1], self.output_dim]), name="dense_weight")
        self.bias = tf.Variable(tf.random.normal([self.output_dim]), name="bias_weight")
        super(MyLayer, self).build(input_shape)

    def call(self, input_tensor):
        """
        call
        :param input_tensor:
        :return:
        """
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

print(model.summary())
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_xs (InputLayer)        [(None, 4)]               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                160
# _________________________________________________________________
# my_layer (MyLayer)           (None, 32)                1056
# _________________________________________________________________
# my_layer_1 (MyLayer)         (None, 48)                1584
# _________________________________________________________________
# dense_2 (Dense)              (None, 64)                3136
# _________________________________________________________________
# predictions (Dense)          (None, 3)                 195
# =================================================================
# Total params: 6,131
# Trainable params: 6,131
# Non-trainable params: 0

