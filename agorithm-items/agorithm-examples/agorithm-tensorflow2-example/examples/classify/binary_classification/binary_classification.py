# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         binary_classification
# Description:  pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --ignore-install tensorflow_hub
#               pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --ignore-install tensorflow_datasets
# Author:       orange
# Date:         2021/5/4
# -------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf    # 2.0.0-beta1
import tensorflow_hub as hub   # 0.5.0
import tensorflow_datasets as tfds

import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images.shape       # (60000, 28, 28)
len(train_labels)        # 60000
train_labels             # ([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
test_images.shape        # (10000, 28, 28)
len(test_labels)         # 10000

# 预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 搭建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
# 模型准备训练前，在模型编译(Compile)时还需要设置一些参数。
# Loss function - 损失函数，训练时评估模型的正确率，希望最小化这个函数，往正确的方向训练模型。
# Optimizer - 优化器算法，更新模型参数的算法。
# Metrics - 指标，用来监视训练和测试步数，下面的例子中使用accuracy，即图片被正确分类的比例。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# 训练神经网络，通常有以下几个步骤。
# 传入训练数据，train_images和train_labels。
# 训练模型去关联图片和标签。
# 模型对测试集test_images作预测，并用test_labels验证预测结果。
model.fit(train_images, train_labels, epochs=10)

# 评估准确率
# 接下来，看看在测试集中表现如何？
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
# 10000/10000 [========] - 0s 37us/sample - loss: 0.3610 - accuracy: 0.8777
# Test accuracy: 0.8777

# 预测
# 使用predict函数进行预测
predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0]) # 9
test_labels[0] # 9
