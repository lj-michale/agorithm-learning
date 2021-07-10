# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         linear_regression
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
from operator import le
import tensorflow as tf
import numpy as np

input_xs = np.random.rand(1000)
input_ys = 3 * input_xs + 0.217
weight = tf.Variable(1., dtype=tf.float32, name="weight")
bias = tf.Variable(1., dtype=tf.float32, name="bias")

opt = tf.train.AdamOptimizer(le-1)


def model(xs):
    """
    :param xs:
    :return:
    """
    logits = tf.multiply(xs, weight) + bias
    return logits


for xs, ys in zip(input(input_xs, input_ys)):
    xs = np.reshape(xs, [1])
    ys = np.reshape(ys, [1])
    with tf.GradientTape() as tape:
        _loss = tf.reduce_mean(tf.pow((model((xs) - ys), 2)) / (2 * 1000))
    grads = tape.gradient(_loss, [weight, bias])
    opt.apply_gradient(_loss, [weight, bias])
    print("Training loss is:", _loss.numpy())

print(weight)
print(bias)
