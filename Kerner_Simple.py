from sklearn.neighbors import kde
import numpy as np

X = np.array([[10,20,30,40],[-2,5,-4,5],[3,1,5,6],[5,20,3,40],[10,20,30,40]])
kde = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
print(kde.score_samples([1,1,1,1]))
print(np.exp(kde.score_samples([1,1,1,1])))
print(np.exp(kde.score_samples([22,2,2,2])))  #值可能大于1  由正态分布的性质决定的
print(np.exp(kde.score_samples([[10,20,30,40],[-2,5,-4,2],[3,1,6,3]])))