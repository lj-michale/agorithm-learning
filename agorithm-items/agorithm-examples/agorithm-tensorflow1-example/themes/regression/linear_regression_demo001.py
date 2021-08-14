# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         linear_regression_demo001
# Description:
# Author:       orange
# Date:         2021/8/14
# -------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义模型超参数
learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.random(*x_train.shape) * 0.33

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w):
    """
    定义模型： y = w:X
    :param X:
    :param w:
    :return:
    """
    return tf.multiply(X, w)


w = tf.Variable(0.0, name="weight")

y_model = model(X, w)
cost = tf.square(Y - y_model)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X:x, Y:y})

w_val = sess.run(w)

sess.close

plt.scatter(x_train, y_train)
y_learned = x_train * w_val
plt.plot(x_train, y_learned, 'r')
plt.show()