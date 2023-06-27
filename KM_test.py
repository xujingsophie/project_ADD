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
from sklearn.cluster import DBSCAN

# data = np.loadtxt(open('E:\异常检测\小区分场景聚类\\bj_data\\KM_dealNULL.csv',encoding='utf-8'),delimiter=',',skiprows=1,usecols=(2,3,4,5,6,7))
# data = pd.read_csv(open('E:\异常检测\小区分场景聚类\\bj_data\\KM_dealNULL.csv'), 'r', encoding='utf-8',sep = ',')

# data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
# f = open('E:/异常检测/小区分场景聚类/bj_data/KM_dealNULL2.csv','rb')
# data = f.read()
# print(chardet.detect(data))


f = csv.reader(open('E:/异常检测/小区分场景聚类/bj_data/HN_HW4215_4.0 - IRsum.csv', 'r')) #, encoding='utf-8'))
data = []
for stu in f:
    data.append(stu)

data1 = np.array(data)
data2 = data1[1:,:]


# kmeans聚类
#假如我要构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters=2)#构造聚类器
estimator.fit(data2)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和


# %%
# 加载数据，显示数据
# digits = datasets.load_digits(n_class=4)
X = data2
y = label_pred
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


# #%%
# #随机映射
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding_2d(X_projected, "Random Projection")

# # %%
# # PCA
# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)
# plot_embedding_2d(X_pca[:, 0:2], "PCA 2D")
# plot_embedding_3d(X_pca, "PCA 3D (time %.2fs)" % (time() - t0))

# #%%
# #LDA
# print("Computing LDA projection")
# X2 = X.copy()
# X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
# t0 = time()
# X_lda = LinearDiscriminantAnalysis(n_components=3).fit_transform(X2, y)
# plot_embedding_2d(X_lda[:,0:2],"LDA 2D" )
# plot_embedding_3d(X_lda,"LDA 3D (time %.2fs)" %(time() - t0))


# %%
# Isomap
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding_2d(X_iso, "Isomap (time %.2fs)" % (time() - t0))

#
#%%
#standard LLE
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding_2d(X_lle,"Locally Linear Embedding (time %.2fs)" %(time() - t0))
#
#
#%%
#modified LLE
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding_2d(X_mlle,"Modified Locally Linear Embedding (time %.2fs)" %(time() - t0))

#
# #%%
# # HLLE
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian') # method='hessian'
# t0 = time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding_2d(X_hlle,"Hessian Locally Linear Embedding (time %.2fs)" %(time() - t0))

# #%%
# # LTSA
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding_2d(X_ltsa,"Local Tangent Space Alignment (time %.2fs)" %(time() - t0))

# # %%
# # MDS
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding_2d(X_mds,"MDS (time %.2fs)" %(time() - t0))

# #%%
# # Random Trees
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)
# plot_embedding_2d(X_reduced,"Random Trees (time %.2fs)" %(time() - t0))


# #%%
# # Spectral
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)
# plot_embedding_2d(X_se,"Spectral (time %.2fs)" %(time() - t0))
#
# # %%
# # t-SNE
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
# t0 = time()
# X_tsne = tsne.fit_transform(X)
# print(X_tsne.shape)
# plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")
# # plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))

plt.show()


y_result = list(label_pred)
y_result.insert(0,'聚类结果')
X_save = np.column_stack((data,y_result))

lines_2 = pd.DataFrame(X_save)
lines_2.to_csv("E:/异常检测/小区分场景聚类/bj_data/HN_HW_KM_RESULT1009.csv.csv",header = 0,index = 0) # 不保留行索引和列名
