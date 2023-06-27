from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
x1 = iris.data

# 目标向量
x2 = iris.target

# #方差选择法，返回值为特征选择后的数据
# #参数threshold为方差的阈值
# y1 = VarianceThreshold(threshold=0.5).fit_transform(iris.data)

# #选择K个最好的特征，返回选择特征后的数据
# #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# #参数k为选择的特征个数
# SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# #选择K个最好的特征，返回选择特征后的数据
# y2 = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

#GBDT作为基模型的特征选择
y3 = SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
