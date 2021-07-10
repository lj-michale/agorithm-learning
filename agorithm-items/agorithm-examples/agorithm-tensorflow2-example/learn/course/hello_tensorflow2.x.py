# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         hello_tensorflow2.x
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf

a = tf.constant(1.) + tf.constant(1.)
text = tf.constant("hello tensorflow 2.x")

print(a)
print(text)