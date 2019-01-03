#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
function: prepare data, 去除特殊符号, 以及特殊字符替换
"""
import re

import pickle
from tqdm import tqdm   # 进度条


def make_split(line):
    # 进行切分合并
    # print(re.match(r'.*([，…?!\.,!？])$', ''.join(line)))
    if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
        return []
    return [', ']


def good_line(line):
    """
    判断是不是有用的句子
    :param line:
    :return:
    """
    new_line = re.findall(r'[a-zA-Z0-9]', ''.join(line))
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        # 如果出现字母或者数字, 将其替换为空字符串, 并加入句子, 如果句子长度大于2, 则不是一个好的句子
        return False
    return True


def regular(sen):
    sen = re.sub(r'\.{3, 100}', '...', sen)  # 句子中连着出现.最少3次, 最多100次, 将其替换为...
    sen = re.sub(r'...{2, 100}', '...', sen)  # 处理......
    sen = re.sub(r'[,]{1, 100}', '，', sen)    # 处理中,
    sen = re.sub(r'[\.]{1,100}', '。', sen)   # 处理.
    sen = re.sub(r'[\?]{1,100}', '？', sen)   # 处理?
    sen = re.sub(r'[!]{1,100}', '！', sen)   # 处理!
    return sen


def main(limit=20, x_limit=3, y_limit=6):
    from word_sequence import WordSequence
    print('extract lines')  # 开始解压文件, 处理好的数据集
    groups, group  = list(), list()
    with open('dgk_shooter_min.conv', 'r', errors='ignore', encoding='utf-8') as fp:
        for line in tqdm(fp):       # tqdm 进度条
            if line.startswith('M'):
                line = line.replace('\n', '')
                if '/' in line:
                    line = line[2:].split('/')
                else:
                    line = list(line[2:])
                line = line[:-1]
                group.append(list(regular(''.join(line))))
            else:
                if group:
                    groups.append(group)
                    group = list()
        if group:
            groups.append(group)
            group = []
    print('extract group')

    # 定义输入和输出, 问答对的处理
    x_data = list()
    y_data = list()
    for group in tqdm(groups):
        for i, line in enumerate(group):   # 获取三行数据, 并赋值
            last_line = None
            if i > 0:
                last_line = group[i-1]
                if not good_line(line=last_line):
                    last_line = None
            next_line = None
            if i < len(group) - 1:
                next_line = group[i+1]
                if not good_line(line=next_line):
                    next_line = None
            next_next_line = None
            if i < len(group) - 2:
                next_next_line = group[i+2]
                if not good_line(line=next_next_line):
                    next_next_line = None
            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:
                x_data.append(last_line+make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)
    print(len(x_data), len(y_data))

    # 设置问和答

    for ask, answer in zip(x_data[:20], y_data[:20]):   # 只取前20个字符
        print(''.join(ask))
        print(''.join(answer))
        print('-'*20)

    data = list(zip(x_data, y_data))  # 将数据打包
    data = [
        (x, y)
        for x, y in data
        if len(x) < limit and len(y) < limit and len(y) >= y_limit and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)
    print('fit word_sequence')
    ws_input = WordSequence()
    ws_input.fit(x_data+y_data)
    print('dump')
    pickle.dump((x_data, y_data),
                open('chatbot.pkl', 'wb'))
    pickle.dump(ws_input, open('ws.pkl', 'wb'))

    print('done')


if __name__ == '__main__':
    # print(good_line(line='你好你发多少发放的safasdgf'))
    main()