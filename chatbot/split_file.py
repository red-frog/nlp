#!/usr/bin/env python3
# -*- coding:utf-8 -*-


fb = open('xiaohuangji.conv', 'a')

with open('xiaohuangji50w_fenciA.conv', 'r') as f:
    j = 0
    for i in f.readlines():
        print(i.strip())
        fb.write(i.strip())
        fb.write('\n')
        j += 1
        if j == 200000:
            break