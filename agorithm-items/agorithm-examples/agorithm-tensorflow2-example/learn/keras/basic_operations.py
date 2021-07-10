# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         basic_operations
# Description:
# Author:       orange
# Date:         2021/6/1
# -------------------------------------------------------------------------------

from __future__ import print_function

# import tensorflow  as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)
print("a:{} b:{}".format(a, b))

# Launch the default graph.
# with tf.compat.v1.Session() as sess:
#     print("a=2, b=3")
#     print("Addition with constants: %i" % sess.run(a+b))
#     print("Multiplication with constants: %i" % sess.run(a*b))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
# with tf.compat.v1.Session() as sess:
#     # Run every operation with variable input
#     print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
#     print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])












































