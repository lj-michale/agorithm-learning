# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         main
# Description:
# Author:       orange
# Date:         2021/7/17
# -------------------------------------------------------------------------------

# ===================== set random  ===========================
import numpy as np
import tensorflow as tf
import random as rn

from learn.course.course003.textcnn import TextCNN

np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)
# =============================================================

import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Bulid Model...')
        self.create_model()

    def create_model(self):
        model = TextCNN(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         kernel_sizes=[2,3,5],
                         kernel_regularizer=None,
                         last_activation='softmax')
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        model.build_graph(input_shape=(None, maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\course003\\logs\\TextCNN-epoch-5', checkpoint_path="E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\course003\\save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_accuracy',
                                             mode='max',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1,
                                             period=2,
                                             )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/TextCNN-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        print(x_val)
        print(y_val)
        self.model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=2,
                  callbacks=self.callback_list,
                  validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)

# ================  params =========================
class_num = 2
maxlen = 400
embedding_dims = 200
epochs = 10
batch_size = 128
max_features = 5000

MODEL_NAME = 'TextCNN-epoch-10-emb-200'

use_early_stop=True
tensorboard_log_dir = 'E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\course003\\logs\\{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = 'E:\\OpenSource\\GitHub\\agorithm-learning\\agorithm-items\\agorithm-examples\\agorithm-tensorflow2-example\\learn\\course\\course003\\save_model_dir\\'+MODEL_NAME+'\\cp-{epoch:04d}.ckpt'
#  ====================================================================

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir, checkpoint_path=checkpoint_path)
model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
print('Test...')
result = model_hepler.model.predict(x_test)
test_score = model_hepler.model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print("test loss:", test_score[0], "test accuracy", test_score[1])


model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.load_model(checkpoint_path=checkpoint_path)
# 重新评估模型
loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))