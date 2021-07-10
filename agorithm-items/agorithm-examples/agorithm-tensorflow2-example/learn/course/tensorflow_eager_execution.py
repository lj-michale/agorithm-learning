# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         tensorflow_eager_execution
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np

data = tf.constant([1, 2])

print(data)
# tf.Tensor([1 2], shape=(2,), dtype=int32)
# Tensor格式 具体数值[1, 2] ,维度大小2，数值类型int32

print(data.numpy())
# [1 2]
# Tensor数据被转化为常用的Numpy数据格式，即常数格式
