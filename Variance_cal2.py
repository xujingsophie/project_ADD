import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import time
from sklearn.preprocessing import Imputer


data = np.loadtxt(open('C:/Users/Jean/Desktop/boston/KPI2/npstoAI.csv','rb'),delimiter=",",skiprows=1)
# data = pd.read_csv('C:/Users/Jean/Desktop/boston/KPI2/npstoAI.csv') # usecols=

# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# data2 = pd.read_csv('C:/Users/Jean/Desktop/boston/KPI2/npstoAI2.csv') # usecols=
# print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# data = data1.append(data2)
# data = pd.read_csv('C:/Users/Jean/Desktop/boston/KPI2/new_biaozhun2.csv') # usecols=
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

ikl = IsolationForest(bootstrap=False)
# ikl = IsolationForest(bootstrap=False,contamination=0.01,n_estimators=256)

# 训练
ikl.fit(data)
print("train is okay")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# 预测
# iris_train = ikl.predict(data)
# print("predict is okay")
iris_score = ikl.decision_function(data)
print("score is okay")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# 预测分数 分越低越可能是异常
# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = 1 - min_max_scaler.fit_transform(iris_score)
X_test = np.array(X_train_minmax)

X_save = np.column_stack((data,X_test))

np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_nps.csv', X_save, fmt="%f", delimiter = ',')
# np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_compare1.csv', iris_train, delimiter = ',')
