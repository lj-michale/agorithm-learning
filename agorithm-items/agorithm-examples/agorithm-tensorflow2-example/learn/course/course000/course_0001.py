# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         course_0001
# Description:
# Author:       orange
# Date:         2021/7/22
# -------------------------------------------------------------------------------

import tensorflow as tf


tf.__version__
print("tensorfow的版本: %s" %(tf.__version__))

v = tf.Variable([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维变量
print("v的值: %s", v)

c = tf.constant([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维常量
print("c的值: %s", c)

# tf.zeros：新建指定形状且全为 0 的常量 Tensor
# tf.zeros_like：参考某种形状，新建全为 0 的常量 Tensor
# tf.ones：新建指定形状且全为 1 的常量 Tensor
# tf.ones_like：参考某种形状，新建全为 1 的常量 Tensor
# tf.fill：新建一个指定形状且全为某个标量值的常量 Tensor

c1 = tf.zeros([3, 3])  # 3x3 全为 0 的常量 Tensor
print("c1的值: %s", c1)

# tf.linspace：创建一个等间隔序列。
# tf.range：创建一个数字序列。
c2 = tf.linspace(1.0, 10.0, 5, name="linspace")
print("c2的值： %s", c2)

# Eager Execution
# TensorFlow 2 带来的最大改变之一是将 1.x 的 Graph Execution（图与会话机制）更改为 Eager Execution（动态图机制）。
# 在 1.x 版本中，低级别 TensorFlow API 首先需要定义数据流图，然后再创建 TensorFlow 会话，这一点在 2.0 中被完全舍弃。
# TensorFlow 2 中的 Eager Execution 是一种命令式编程环境，可立即评估操作，无需构建图。
# 所以说，TensorFlow 的张量运算过程可以像 NumPy 一样直观且自然了。接下来，我们以最简单的加法运算为例：
a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3])
b = tf.constant([7., 8., 9., 10., 11., 12.], shape=[3, 2])
c3 = tf.linalg.matmul(a, b)  # 矩阵乘法
print("c3的值： %s", c3)

# 自动微分
# 在数学中，微分是对函数的局部变化率的一种线性描述。虽然微分和导数是两个不同的概念。
# 但是，对一元函数来说，可微与可导是完全等价的。如果你熟悉神经网络的搭建过程，应该明白梯度的重要性。
# 而对于复杂函数的微分过程是及其麻烦的，为了提高应用效率，大部分深度学习框架都有自动微分机制。
# TensorFlow 中，你可以使用 tf.GradientTape 跟踪全部运算过程，以便在必要的时候计算梯度。
w = tf.Variable([1.0])  # 新建张量
with tf.GradientTape() as tape:  # 追踪梯度
    loss = w * w
grad = tape.gradient(loss, w)  # 计算梯度
print("grad的值： %s", grad)


# 常用模块
# 上面，我们已经学习了 TensorFlow 核心知识，接下来将对 TensorFlow API 中的常用模块进行简单的功能介绍。
# 对于框架的使用，实际上就是灵活运用各种封装好的类和函数。由于 TensorFlow API 数量太多，迭代太快，所以大家要养成随时 查阅官方文档 的习惯。
# tf.：包含了张量定义，变换等常用函数和类。
# tf.data：输入数据处理模块，提供了像 tf.data.Dataset 等类用于封装输入数据，指定批量大小等。
# tf.image：图像处理模块，提供了像图像裁剪，变换，编码，解码等类。
# tf.keras：原 Keras 框架高阶 API。包含原 tf.layers 中高阶神经网络层。
# tf.linalg：线性代数模块，提供了大量线性代数计算方法和类。
# tf.losses：损失函数模块，用于方便神经网络定义损失函数。
# tf.math：数学计算模块，提供了大量数学计算函数。
# tf.saved_model：模型保存模块，可用于模型的保存和恢复。
# tf.train：提供用于训练的组件，例如优化器，学习率衰减策略等。
# tf.nn：提供用于构建神经网络的底层函数，以帮助实现深度神经网络各类功能层。
# tf.estimator：高阶 API，提供了预创建的 Estimator 或自定义组件。

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
np.argmax(predictions[0])
test_labels[0]


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Use the trained model
# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])

