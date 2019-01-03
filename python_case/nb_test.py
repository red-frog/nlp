#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# 1. 搜集数据
# 2. 处理数据

import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import time


def pre_process(path):
    """
    预处理, 切词
    :param path: 文件路径
    :return:
    """
    text_with_space = ""
    with open(path, 'r', encoding="utf-8") as f:
        textfile = f.read()
        textcut = jieba.cut(textfile)
        for word in textcut:
            text_with_space += word + " "
    return text_with_space


def load_train_data_set(path, classtag):
    allfiles = os.listdir(path)
    processed_text_set = list()
    all_class_tags = list()
    for file_name in allfiles:
        print(file_name)
        path_name = path+'/'+file_name
        processed_text_set.append(pre_process(path_name))
        all_class_tags.append(classtag)
    return processed_text_set, all_class_tags  # 返回数据集和标签号


processed_textdata1, class1 = load_train_data_set(path="./dataset/train/hotel", classtag="宾馆")
processed_textdata2, class2 = load_train_data_set(path="./dataset/train/travel", classtag="旅游")

train_data = processed_textdata1 + processed_textdata2   # 对训练数据进行整合

classtags_list = class1 + class2  # 标签整合

count_vector = CountVectorizer()   # 统计词向量
vecot_matrix = count_vector.fit_transform(train_data)  # 转换成词频

# TFIDF(度量模型, 文本主题模型) 特征工程统计
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vecot_matrix)

# 朴素贝叶斯进行训练

clf = MultinomialNB().fit(train_tfidf, classtags_list)

# 对训练后结果进行测试

path = "./dataset/test/travel"
allfiles = os.listdir(path)
hotel = 0
travel = 0
for file_name in allfiles:
    path_name = path + "/" + file_name
    new_count_vector = count_vector.transform([pre_process(path_name)])  # 预处理
    new_tfidf = TfidfTransformer(use_idf=False).fit_transform(new_count_vector)  # 提取特征
    # 进行预测
    predict_result = clf.predict(new_tfidf)
    print(str(predict_result)+file_name)

    # 统计正确率
    if predict_result == '宾馆':
        hotel += 1
    if predict_result == '旅游':
        travel += 1

print('宾馆' + str(hotel))
print('旅游' + str(travel))