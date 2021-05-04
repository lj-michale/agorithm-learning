# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         classify_structured_data
# Description:  特征工程结构化数据分类
#               pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --ignore-install sklearn
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()