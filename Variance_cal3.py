import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import time
from sklearn.preprocessing import Imputer

data = np.loadtxt(open('E:\异常检测\小区分场景聚类\\bj_data\\HN_HW4215_4.0 - IRsum.csv','rb'),delimiter=",",skiprows=1)#usecols=(1,2,3,4,5,6,7))

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

ikl = IsolationForest(bootstrap=False,contamination=0.01)
# ikl = IsolationForest(bootstrap=False,contamination=0.01,n_estimators=256)

ikl.fit(data)
print("train is okay")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# 预测
# 预测分数 分越低越可能是异常
iris_score = ikl.decision_function(data)
iris_train = ikl.predict(data)
print("score is okay")
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# 归一化
iris_score = iris_score.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = 1 - min_max_scaler.fit_transform(iris_score)
X_test = np.array(X_train_minmax)

# 把最后一列加到原文件上
X_save = np.column_stack((data,X_test))
# save_row = np.array([[AA,AA,AA,AA]])
# X_save = np.insert(X_save1,0,values=['000','000','000','000','000','000','000','000'],axis=0)

np.savetxt('E:\异常检测\小区分场景聚类\\bj_data\\HN_HW4215_4.0 - RESULT.csv', X_save, fmt="%f", delimiter = ',')
# np.savetxt('C:/Users/Jean/Desktop/boston/KPI2/new_compare1.csv', iris_train, delimiter = ',')
