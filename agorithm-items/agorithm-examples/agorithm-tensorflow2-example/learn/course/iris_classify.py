# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         iris_classify
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np

from sklearn.datasets import load_iris

iris_classify_model_path = "E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\models\\iris_classify\\clssify_save_model.h5"

data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)

iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))
iris_data = tf.data.Dataset.from_tensor_slices(iris_data).batch(50)
iris_target = tf.data.Dataset.from_tensor_slices(iris_target).batch(50)

# 输入端
inputs = tf.keras.layers.Input(shape=(4))

# 中间层
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)

predictions = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
opt = tf.optimizers.Adam(1e-3)

for epoch in range(1000):
    for _data, label in zip(iris_data, iris_target):
        with tf.GradientTape() as tape:
            logits = model(_data)
            loss_value = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=label, y_pred=logits))
            grads = tape.gradient(loss_value, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            print("Training loss is:", loss_value.numpy())

model.save(iris_classify_model_path)

new_model = tf.keras.models.load_model(iris_classify_model_path)
new_prediction = new_model.predict(iris_data)
print(tf.argmax(new_prediction, axis=-1))


