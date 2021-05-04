# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         imdb_text_classification
# Description:
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf             # 2.0.0-beta1
import tensorflow_hub as hub        # 0.5.0
import tensorflow_datasets as tfds

# 进一步划分训练集。
# 60%(15,000)用于训练，40%(10,000)用于验证(validation)。
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

train_labels_batch
# <tf.Tensor: id=221, shape=(10,), dtype=int64, numpy=array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0])>

# 搭建模型
# 神经网络需要堆叠多层，架构上需要考虑三点。
# 1. 文本怎么表示？
# 2. 模型需要多少层？
# 3. 每一层多少个_隐藏节点_
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# 搭建完整的神经网络模型
# 第一层是 TensorFlow Hub 层，将句子转换为 tokens，然后映射每个 token，并组合成最终的向量。输出的维度是：句子个数 * 嵌入维度(20)。
# 接下来是全连接层(Full-connected, FC)，即Dense层，16个节点。
# 最后一层，也是全连接层，只有一个节点。使用sigmoid激活函数，输出值是float，范围0-1，代表可能性/置信度。
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 损失函数和优化器
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
# loss: 0.314
# accuracy: 0.866

