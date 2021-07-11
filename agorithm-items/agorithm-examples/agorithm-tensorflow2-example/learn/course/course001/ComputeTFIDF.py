# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         ComputeTFIDF
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
import math

from learn.course.course001.Word2Vec import agnews_text


def idf(corpus):
    idfs = {}
    d = 0.0
    # 统计词出现次数
    for doc in corpus:
        d += 1
        counted = []
        for word in doc:
            if not word in counted:
                counted.append(word)
                if word in idfs:
                    idfs[word] += 1
                else:
                    idfs[word] = 1
    # 计算每个词逆文档值
    for word in idfs:
        idfs[word] = math.log(d/float(idfs[word]))
    return idfs

# 获取计算好的文本中每个词的idf词频，agnews_text是经过处理后的语料库文档，在数据清洗一节中有详细介绍
idfs = idf(agnews_text)
# 获取文档集中每个文档
for text in agnews_text:
    word_tfidf = {}
    for word in text:      # 依次获取每个文档中的每个词
        if word in word_tfidf:		# 计算每个词的词频
            word_tfidf[word] += 1
        else:
            word_tfidf[word] = 1
    for word in word_tfidf:
        word_tfidf[word] *= idfs[word]   # word_tfidf为计算后的每个词的TFIDF值

    values_list = sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True)  # 按value排序
    # 生成排序后的单个文档
    values_list = [value[0] for value in values_list]
