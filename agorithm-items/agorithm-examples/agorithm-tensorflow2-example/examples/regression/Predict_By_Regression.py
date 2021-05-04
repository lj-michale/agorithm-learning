# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Predict_By_Regression
# Description:  pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --ignore-install seaborn
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Auto MPG 数据集
# 下载数据集到本地
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = keras.utils.get_file("auto-mpg.data", url)

# 使用Pandas读取数据
column_names = ['MPG','气缸','排量','马力','重量','加速度', '年份', '产地']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# 查看前3条数据
dataset.head(3)
print(dataset.head(3))

dataset.isna().sum()
print(dataset.isna().sum())

dataset = dataset.dropna()

origin = dataset.pop('产地')
dataset['美国'] = (origin == 1)*1.0
dataset['欧洲'] = (origin == 2)*1.0
dataset['日本'] = (origin == 3)*1.0
# 看一看转换后的结果
print(dataset.head(3))

# 训练集 80%， 测试集 20%
# 划分训练集与测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset)
print(test_dataset)

# 解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

sns.pairplot(train_dataset[["MPG", "气缸", "排量", "重量"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
print(train_stats)

# 分离 label
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 归一化数据
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 搭建模型
# 我们的模型包含2个全连接的隐藏层构成，输出层返回一个连续值。
def build_model():
  input_dim = len(train_dataset.keys())
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[input_dim,]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  model.compile(loss='mse', metrics=['mae', 'mse'],
                optimizer=tf.keras.optimizers.RMSprop(0.001))
  return model

model = build_model()
# 打印模型的描述信息，每一层的大小、参数个数等
model.summary()
print(model.summary())

# 训练模型
import sys

EPOCHS = 1000


class ProgressBar(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):
        # 显示进度条
        self.draw_progress_bar(epoch + 1, EPOCHS)

    def draw_progress_bar(self, cur, total, bar_len=50):
        cur_len = int(cur / total * bar_len)
        sys.stdout.write("\r")
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
        sys.stdout.flush()


history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[ProgressBar()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail(3)
print(hist.tail(3))


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('metric - MSE')
    plt.plot(hist['epoch'], hist['mse'], label='训练集')
    plt.plot(hist['epoch'], hist['val_mse'], label='验证集')
    plt.ylim([0, 20])
    plt.legend()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('metric - MAE')
    plt.plot(hist['epoch'], hist['mae'], label='训练集')
    plt.plot(hist['epoch'], hist['val_mae'], label='验证集')
    plt.ylim([0, 5])
    plt.legend()


plot_history(history)
print(plot_history(history))

model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0,
                    callbacks=[early_stop, ProgressBar()])
plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("测试集平均绝对误差(MAE): {:5.2f} MPG".format(mae))
# 测试集平均绝对误差(MAE):  1.90 MPG

# 预测
test_pred = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_pred)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])






















