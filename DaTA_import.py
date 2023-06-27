import csv
import numpy as np
from sklearn.neighbors import kde
from sklearn import preprocessing
import sys

csv_reader = csv.reader(open('C:/Users/x1carbon/Desktop/指标异常检测/test_file.csv'))

for row in csv_reader:
    print(row)

arr = np.loadtxt('C:/Users/x1carbon/Desktop/指标异常检测/test_file2.csv',delimiter=',')
X = np.array(arr)
kde = kde.KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
result=np.exp(kde.score_samples(X))  # 对log函数进行计算
print(result)

# 归一化处理
X_train = np.array(result)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)

