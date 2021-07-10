# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         course_softmax
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
import math

import numpy as np


def softmax(inMatrix):
    """
    softmax
    :param inMatrix:
    :return:
    """
    m, n = np.shape(inMatrix)
    outMatrix = np.mat(np.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        outMatrix[0, idx] = math.exp(inMatrix[0, idx])
        soft_sum += outMatrix[0, idx]
    for idx in range(0, n):
        outMatrix[0, idx] = outMatrix[0, idx] / soft_sum
    return outMatrix
