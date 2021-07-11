# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         TextRank
# Description:
# Author:       orange
# Date:         2021/7/10
# -------------------------------------------------------------------------------、


class TextRank_score:
    def __init__(self,agnews_text):
        self.agnews_text = agnews_text
        self.filter_list = self.__get_agnews_text()
        self.win = self.__get_win()
        self.agnews_text_dict = self.__get_TextRank_score_dict()

    def __get_agnews_text(self):
        sentence = []
        for text in self.agnews_text:
            for word in text:
                sentence.append(word)
        return sentence

    def __get_win(self):
        win = {}
        for i in range(len(self.filter_list)):
            if self.filter_list[i] not in win.keys():
                win[self.filter_list[i]] = set()
            if i - 5 < 0:
                lindex = 0
            else:
                lindex = i - 5
            for j in self.filter_list[lindex:i + 5]:
                win[self.filter_list[i]].add(j)
        return win
    def __get_TextRank_score_dict(self):
        time = 0
        score = {w: 1.0 for w in self.filter_list}
        while (time < 50):
            for k, v in self.win.items():
                s = score[k] / len(v)
                score[k] = 0
                for i in v:
                    score[i] += s
            time += 1
        agnews_text_dict = {}
        for key in score:
            agnews_text_dict[key] = score[key]
        return agnews_text_dict

    def __get_TextRank_score(self, text):
        temp_dict = {}
        for word in text:
            if word in self.agnews_text_dict.keys():
                temp_dict[word] = (self.agnews_text_dict[word])
        values_list = sorted(temp_dict.items(), key=lambda word_tfidf: word_tfidf[1],
                             reverse=False)  # 将TextRank数据按重要程度从大到小排序
        return values_list
    def get_TextRank_result(self,text):
        temp_dict = {}
        for word in text:
            if word in self.agnews_text_dict.keys():
                temp_dict[word] = (self.agnews_text_dict[word])
        values_list = sorted(temp_dict.items(), key=lambda word_tfidf: word_tfidf[1], reverse=False)
        value_list = []
        for value in values_list:
            value_list.append(value[0])
        return (value_list)
