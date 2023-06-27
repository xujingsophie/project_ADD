#!/usr/bin/python
# -*- coding:utf-8-*-

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import copy
import csv
import chardet


# data = np.loadtxt(open('E:\异常检测\小区分场景聚类\bj_data\test1.csv','rb'),delimiter=",",skiprows=1,usecols=(1,2,3,4,5,6,7))
# f1 = pd.read_csv("E:/异常检测/小区分场景聚类/bj_data/test1.csv", "r", encoding='GB18030',sep = ",")

# f = open(path,'rb')
# data = f.read()
# print(chardet.detect(data))

f = csv.reader(open("E:/异常检测/小区分场景聚类/bj_data/bjkpi33.csv",'r'))#,encoding='utf-8'))
lines = []
for stu in f:
    lines.append(stu)
    # print(stu)
# lines = f.readlines()  # read all lines
# print(lines)
sam_point = 0
cov_rat = 0
inout_door = 0
core_city = 0
ERAB_drop = 0
PRB_avr = 0
line_1 = lines[0]  # deal the first line
print(len(line_1))
for k in range(len(line_1)):  # get the position of each data
    if line_1[k] == "VoLTE下行平均时延":
        sam_point = k
    if line_1[k] == "PDCCH信道CCE占用率":
        cov_rat = k
    if line_1[k] == "无线接通率":
        inout_door = k
    if line_1[k] == "无线接通率(QCI=1)":
        core_city = k
    if line_1[k] == "E-RAB掉线率(剔除UI原因)":
        ERAB_drop = k
    if line_1[k] == "上行PRB平均利用率":
        PRB_avr = k

lines_1 = copy.copy(lines)  # 复制lines

for index,line in enumerate(lines[1:]):  # 注意index和line的顺序
    # print(line_2[sam_point],line_2[cov_rat])
    # print(index/100)
    # if ('\\N' in line):
    #     # print(lines[index + 1])  # 输出有空字段的行
    #     lines_1.remove(line)
    # if((index%10000) == 0):
    #     print(index)
    if (line[core_city] == '\\N'):
        line[core_city] = 1
    if (line[inout_door] == '\\N'):
        line[inout_door] = 1
    # if ((line[sam_point] or line[cov_rat] or line[core_city] or line[ERAB_drop] \
    #     or line[PRB_avr] or line[inout_door])== '\\N'):  # 或者r'\N'   # and line[cov_rat]
    #     # 如果Volte下行平均时延和PDCCH信道CCE占用率为空，删掉这一行。
    #     print(index)
    #     print(lines[index+1])
    #     lines_1.remove(line)
    # elif (line[inout_door] == '\\N'):
    #     print(lines_1[index+1][inout_door])
    #     lines_1[index][inout_door] = 0
    # elif (line[core_city] == '\\N'):
    #     print(lines_1[index+1][core_city])
    #     lines_1[index][core_city] = 1

lines_1 = copy.copy(lines)

for index,line in enumerate(lines[1:]):  # 注意index和line的顺序
    if ('\\N' in line):
        # print(lines[index + 1])  # 输出有空字段的行
        lines_1.remove(line)
    if((index%10000) == 0):
        print(index)

lines_2 = pd.DataFrame(lines_1)
lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_RESULT.csv",header = 0,index = 0) # 不保留行索引和列名

