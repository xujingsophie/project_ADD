#coding:utf-8

from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection,preprocessing)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
import csv
import pandas as pd

f = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_RESULT1_KM0.csv', 'r')) #, encoding='utf-8'))
data = []
for stu in f:
    data.append(stu[1])

f1 = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_DMselect.csv', 'r')) #, encoding='utf-8'))
data1 = []
for stu1 in f1:
    data1.append(stu1)

data2 = []
for index,i in enumerate(data):  # 筛选两个表中一致的数据
    print(index)
    for j in data1:
        if (j[1] == i):
            data2.append(j)

lines_2 = pd.DataFrame(data2)
lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_IRpre_KM0.csv" ,header = 0,index = 0) # 不保留行索引和列名



