from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import style
from time import time
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import csv
import plotfunc
import pandas as pd

print('start')
f = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/HN_HW4215_4.0 - IRsum.csv', 'r')) #, encoding='utf-8'))
data = []
for stu in f:
    data.append(stu)

data1 = np.array(data)
data2 = data1[1:,0:]
data3 = np.array(data2,dtype= np.float64) # 变换array的数据类型


clf=MeanShift()
'''对样本数据进行聚类'''
predicted=clf.fit_predict(data3)

plotfunc.plotfunc(data3,predicted)

# 把最后一列加到原文件上
y_result = list(predicted)
y_result.insert(0,'聚类结果')
X_save = np.column_stack((data,y_result))
# save_row = np.array([[AA,AA,AA,AA]])
# X_save = np.insert(X_save1,0,values=['000','000','000','000','000','000','000','000'],axis=0)

lines_2 = pd.DataFrame(X_save)
lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/HN_HW_Meanshift_RESULT.csv.csv",header = 0,index = 0) # 不保留行索引和列名
