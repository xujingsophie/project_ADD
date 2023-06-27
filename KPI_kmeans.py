#!/usr/bin/python
# -*- coding:utf-8-*-

import numpy as np
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
import csv
import pandas as pd
import chardet
import plotfunc

# data = np.loadtxt(open('E:\异常检测\小区分场景聚类\\bj_data\\KM_dealNULL3.csv',encoding='GB2312'),delimiter=',',skiprows=1,usecols=(2,3,4,5,6,7))
# data = pd.read_csv('E:\异常检测\小区分场景聚类\\bj_data\\KM_dealNULL3.csv', 'r',delimiter=',',encoding='utf-8')

# data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
# f = open('E:/异常检测/小区分场景聚类/bj_data/KM_dealNULL2.csv','rb')
# data = f.read()
# print(chardet.detect(data))


f = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_RESULT1.csv', 'r')) #, encoding='utf-8'))
data = []
for stu in f:
    data.append(stu)

data1 = np.array(data)
data2 = data1[1:,2:]

# kmeans聚类 观察K的取值
# centerKM = []
# 假如我要构造一个聚类数为3的聚类器
# for i in range(1,10):
#     estimator = KMeans(n_clusters=i,init='k-means++')# 构造聚类器使用k-means++算法
#     estimator.fit(data2)#聚类
#     label_pred = estimator.labels_ #获取聚类标签
#     centroids = estimator.cluster_centers_ #获取聚类中心
#     inertia = estimator.inertia_ # 获取聚类准则的总和
#     centerKM.append(inertia)
#
# x = range(1,10)
# x = np.linspace(0,8,9)
# y = centerKM
# plt.plot(x,y,'-')
# plt.show()

estimator = KMeans(n_clusters=6,init='k-means++')# 构造聚类器使用k-means++算法
estimator.fit(data2)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

# 把最后一列加到原文件上
y_result = list(label_pred)
y_result.insert(0, '聚类结果')
X_save = np.column_stack((data, y_result))

# for i in range(0,5):   # 针对每一类输出小区
#     lines_1 = []
#     lines_1.append(X_save[0])
#     for j in X_save[1:-1]:
#         if (int(j[-1]) == i):   # 注意数据格式
#             lines_1.append(j)
#
#     lines_2 = pd.DataFrame(lines_1)
#     lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_RESULT1_KM%s.csv" % i,header = 0,index = 0) # 不保留行索引和列名


plotfunc.plotfunc(data2,label_pred)