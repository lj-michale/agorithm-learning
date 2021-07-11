# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Word2Vec
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------

import csv
import tools
import numpy as np

from gensim.models import Word2Vec

agnews_label = []
agnews_title = []
agnews_text = []
agnews_train = csv.reader(open("./dataset/train.csv","r"))

for line in agnews_train:
    agnews_label.append(np.float32(line[0]))
    agnews_title.append(tools.text_clear(line[1]))
    agnews_text.append(tools.text_clear(line[2]))

print("开始训练模型")
model = Word2Vec(agnews_text, size=64, min_count=0, window=5, iter=128)
model_name = "corpusWord2Vec.bin"
model.save(model_name)

model = Word2Vec.load('./corpusWord2Vec.bin')
model.train(agnews_title, epochs=model.epochs, total_examples=model.corpus_count)
