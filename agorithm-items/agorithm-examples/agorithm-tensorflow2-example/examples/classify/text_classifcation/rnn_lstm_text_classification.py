# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         rnn_lstm_text_classification
# Description:  RNN LSTM 文本分类
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
# geektutu.com

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import Sequential, layers

ds, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_ds, test_ds = ds['train'], ds['test']

BUFFER_SIZE, BATCH_SIZE = 10000, 64
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.padded_batch(BATCH_SIZE, train_ds.output_shapes)
test_ds = test_ds.padded_batch(BATCH_SIZE, test_ds.output_shapes)

# geektutu.com
tokenizer = info.features['text'].encoder
print ('词汇个数:', tokenizer.vocab_size)

sample_str = 'welcome to geektutu.com'
tokenized_str = tokenizer.encode(sample_str)
print ('向量化文本:', tokenized_str)

for ts in tokenized_str:
    print (ts, '-->', tokenizer.decode([ts]))


# 搭建 RNN 模型
# geektutu.com
model = Sequential([
    layers.Embedding(tokenizer.vocab_size, 64),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history1 = model.fit(train_ds, epochs=3, validation_data=test_ds)
loss, acc = model.evaluate(test_ds)
print('准确率:', acc) # 0.81039

# geektutu.com
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20


def plot_graphs(history, name):
    plt.plot(history.history[name])
    plt.plot(history.history['验证集 - '+ name])
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend([name, '验证集 - ' + name])
    plt.show()


plot_graphs(history1, 'accuracy')

# 添加更多 LSTM 层
# geektutu.com
model = Sequential([
    layers.Embedding(tokenizer.vocab_size, 64),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=3, validation_data=test_ds)
loss, acc = model.evaluate(test_ds)
print('准确率:', acc) # 0.83096%

# geektutu.com
plot_graphs(history, 'accuracy')











