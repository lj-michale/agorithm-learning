# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         TFIDF
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------
import math


class TFIDF_score:
    def __init__(self, corpus, model=None):
        self.corpus = corpus
        self.model = model
        self.idfs = self.__idf()

    def __idf(self):
        idfs = {}
        d = 0.0
        # 统计词出现次数
        for doc in self.corpus:
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
            idfs[word] = math.log(d / float(idfs[word]))
        return idfs

    def __get_TFIDF_score(self, text):
        word_tfidf = {}
        # 依次获取每个文档中的每个词
        for word in text:
            # 计算每个词的词频
            if word in word_tfidf:
                word_tfidf[word] += 1
            else:
                word_tfidf[word] = 1
        for word in word_tfidf:
            # 计算每个词的TFIDF值
            word_tfidf[word] *= self.idfs[word]
        # 将TFIDF数据按重要程度从大到小排序
        values_list = sorted(word_tfidf.items(), key=lambda word_tfidf: word_tfidf[1], reverse=True)
        return values_list

    def get_TFIDF_result(self,text):
        values_list = self.__get_TFIDF_score(text)
        value_list = []
        for value in values_list:
            value_list.append(value[0])
        return (value_list)
