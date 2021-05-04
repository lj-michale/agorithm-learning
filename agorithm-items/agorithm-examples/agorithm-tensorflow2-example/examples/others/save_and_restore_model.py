# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         save_and_restore_model
# Description:  保存与加载模型
#               pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --ignore-install pyyaml
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.datasets import mnist

import os
file_path = os.path.abspath('./mnist.npz')

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data(path=file_path)
train_y, test_y = train_y[:1000], test_y[:1000]
train_x = train_x[:1000].reshape(-1, 28 * 28) / 255.0
test_x = test_x[:1000].reshape(-1, 28 * 28) / 255.0


# 搭建模型
def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

    return model


def evaluate(target_model):
    _, acc = target_model.evaluate(test_x, test_y)
    print("Restore model, accuracy: {:5.2f}%".format(100*acc))


# 自动保存 checkpoints
# 这样做，一是训练结束后得到了训练好的模型，使用得不必再重新训练，二是训练过程被中断，可以从断点处继续训练。
# 设置tf.keras.callbacks.ModelCheckpoint回调可以实现这一点。
# 存储模型的文件名，语法与 str.format 一致
# period=10：每 10 epochs 保存一次
checkpoint_path = "file:\\\E:\OpenSource\GitHub\agorithm-learning\agorithm-items\agorithm-examples\agorithm-tensorflow2-example\docs\training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, period=10)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_x, train_y, epochs=50, callbacks=[cp_callback],
          validation_data=(test_x, test_y), verbose=0)

# 加载权重：
latest = tf.train.latest_checkpoint(checkpoint_dir)
# 'training_2/cp-0050.ckpt'
model = create_model()
model.load_weights(latest)
evaluate(model)

# 手动保存权重
# 手动保存权重
model.save_weights('./checkpoints/mannul_checkpoint')
model = create_model()
model.load_weights('./checkpoints/mannul_checkpoint')
evaluate(model)

# 保存整个模型
# HDF5
# 直接调用model.save即可保存为 HDF5 格式的文件。
model.save('my_model.h5')
new_model = models.load_model('my_model.h5')
evaluate(new_model)

# saved_model
# 保存为saved_model格式。
import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model, saved_model_path)

new_model.compile(optimizer=model.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
evaluate(new_model)

# TensorFlow 中还有其他的方式可以保存模型。
# Saving in eager eager 模型保存模型
# Save and Restore – low-level 的接口。










