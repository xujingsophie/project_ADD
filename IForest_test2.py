import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing

ilf = IsolationForest(n_estimators=100)
data = pd.read_csv('C:/Users/Jean/Desktop/boston//bostonprice_test.csv')
data = data.fillna(0)
# 选取特征，不使用标签(类型)
# X_cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

print(data.shape)

# 训练
clf = IsolationForest(max_samples=100)

# 预测
clf.fit(data) # 输出正常还是异常
y_pred_train = clf.predict(data) # 输出分数 分越低越异常



np.savetxt('C:/Users/Jean/Desktop/boston//new.csv', y_pred_train, delimiter = ',')