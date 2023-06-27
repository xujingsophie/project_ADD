import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing


data = pd.read_csv('C:/Users/Jean/Desktop/boston/KPI2/HN_HW4215_4.0.csv') # usecols=
# data = pd.read_csv('C:/Users/Jean/Desktop/boston/KPI2/new_biaozhun2.csv') # usecols=

ikl = IsolationForest(bootstrap=False)
# ikl = IsolationForest(bootstrap=False,contamination=0.01,n_estimators=256)

data_record = np.zeros(len(data))
data_record2 = np.zeros(len(data))

num_train = 5 # 训练次数
num_error = 1000 # 异常数据个数

for i in range(num_train):
    # 训练
    ikl.fit(data)
    # 预测
    iris_train = ikl.predict(data)
    iris_score = ikl.decision_function(data)

    # 预测分数 分越低越可能是异常

    # for i in range(5):
    # iris_train = ikl.predict(data)
    # iris_sum = iris_sum + iris_train

    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = 1 - min_max_scaler.fit_transform(iris_score)
    X_test = np.array(X_train_minmax)

    data_index = X_test.argsort()[-num_error:][::-1] # 找出矩阵的前num_error个最大的元素的index:

    for i in data_index:  # 对前num_error个异常值进行记录
        data_record[i] += X_test[i]
        data_record2[i] += 1

    # for i in range(len(X_test)): # 对前100个异常值进行统计
    #     if (X_test[i]>=0.8):
    #         data_record[i]=data_record[i]+1

# for i in range(len(data_record2)):  # 对总个数小于num_train的清零
#     if (data_record2[i]!=num_train):
#         data_record[i] = 0

np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_compare1.csv', data_record/num_train, fmt="%f", delimiter = ',')
np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_compare2.csv', data_record2, fmt="%f", delimiter = ',')
# np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_compare1.csv', iris_train, delimiter = ',')

# data_test3 = np.row_stack((['AA','AA','AA','AA','AA','AA','AA'],data_test1)) #添加首行

# 归一化
# for i in range(len_test):
#     X_train = np.array(data)[:,i]
#     min_max_scaler = preprocessing.MinMaxScaler()
#     X_train_minmax = 1-min_max_scaler.fit_transform(X_train)
#     X_test = np.array(X_train_minmax)
#     data_test1[:,i] = X_test