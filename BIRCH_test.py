
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
from sklearn import metrics
from time import time
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)

# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
# X,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
#                   random_state =9)
# plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()

f = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_5万.csv', 'r')) #, encoding='utf-8'))
data = []
for stu in f:
    data.append(stu)

data1 = np.array(data)
data2 = data1[1:,2:]
data3 = np.array(data2,dtype= np.float64) # 变换array的数据类型


# y_pred = Birch(n_clusters = None).fit_predict(X)
y_pred = Birch(n_clusters = 5, threshold=0.1, branching_factor=45).fit_predict(data3)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(data3, y_pred))

'''
# %%
# 加载数据，显示数据
# digits = datasets.load_digits(n_class=4)
X = data2
y = y_pred
print(X.shape)
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.title('A selection from the 64-dimensional digits dataset')

#LLE,Isomap,LTSA需要设置n_neighbors这个参数
n_neighbors = 30

#%%
# 将降维后的数据可视化,2维
def plot_embedding_2d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],str(y[i]),
        # ax.text(X[i, 0], X[i, 1],str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)

#%%
#将降维后的数据可视化,3维
def plot_embedding_3d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2],str(y[i]),
        # ax.text(X[i, 0], X[i, 1], X[i,2],str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)

# %%
# Isomap
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding_2d(X_iso, "Isomap (time %.2fs)" % (time() - t0))

#
#%%
# #standard LLE
# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
# t0 = time()
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding_2d(X_lle,"Locally Linear Embedding (time %.2fs)" %(time() - t0))
# #
#
#%%
#modified LLE
# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding_2d(X_mlle,"Modified Locally Linear Embedding (time %.2fs)" %(time() - t0))
# #

# #%%
# # Random Trees
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)
# plot_embedding_2d(X_reduced,"Random Trees (time %.2fs)" %(time() - t0))
#
#
# %%
# Spectral
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)
# plot_embedding_2d(X_se,"Spectral (time %.2fs)" %(time() - t0))

# %%
# t-SNE
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
# t0 = time()
# X_tsne = tsne.fit_transform(X)
# print(X_tsne.shape)
# plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")
# # plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))

plt.show()
'''

# 把最后一列加到原文件上
y_result = list(y_pred)
y_result.insert(0,'聚类结果')
X_save = np.column_stack((data,y_result))

lines_2 = pd.DataFrame(X_save)
lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/bjkpi33删除空字段_RESULT.csv",header = 0,index = 0) # 不保留行索引和列名
