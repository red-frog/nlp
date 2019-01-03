#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class WordSequence(object):
    """
    将句子编码化, 字典定义及转化
    """
    PAD_TAG = "<pad>"  # 填充标签
    UNK_TAG = "<unk>"  # 未知标签
    START_TAG = '<s>'  # 开始标签
    END_TAG = '</S>'  # 结束标签

    PAD, UNK, START, END = 0, 1, 2, 3

    def __init__(self):
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END
        }
        self.fited = False  # 训练标志, 是否训练

    def to_index(self, word):
        assert self.fited, 'WordSequence 尚未进行 fit 操作'
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):  # 将index转为word
        assert self.fited, 'WordSequence 尚未进行 fit 操作'
        for k, v in self.dict.items():
            if v == index:
                return k

        return WordSequence.UNK_TAG

    def size(self):
        assert self.fited, 'WordSequence 尚未进行 fit 操作'
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        """简单的拟合的过程"""
        assert not self.fited, 'WordSequence 只能fit一次'
        count = dict()
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1
        if min_count:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count:
            count = {k: v for k, v in count.items() if v <= max_count}

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END
        }

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x:x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]  # 字典的切分
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        self.fited = True

    def transform(self, sentence, max_len=None):  # 将句子转成向量
        assert self.fited, 'WordSequence 尚未进行 fit 操作'
        if max_len:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        for index, a in enumerate(sentence):
            if max_len and index <= len(r):
                break
            r[index] = self.to_index(a)
        return np.array(r)

    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False,
                          ignore_start=False, ignore_end=False):  # 将向量转成句子
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)
        return ret


def test():
    ws = WordSequence()
    ws.fit([
        ['你', '好', '啊'],
        ['你', '好', '哦']
    ])
    indice = ws.transform(['我', '们', '好'])
    print(indice)
    back = ws.inverse_transform(indice)
    print(back)


if __name__ == '__main__':
    test()