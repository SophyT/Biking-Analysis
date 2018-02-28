#Affinity Propagation 不需要制定类的数量，受初始中心点和数据影响小，稳定性高于kmeans，但在大数据集上运行速度慢
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import AffinityPropagation

data=pd.read_csv("prodata.csv")

# Affinity Propagation
af = AffinityPropagation(damping=0.87).fit(data)#可调参数dampling,取值(0.5,1),值越大，分出的cluster越多.在该小数据集上，经多次测试发现，当dampling大约取（0.5，0.86），分两类；取[0.9，1）,分11类
numSamples = len(data)
centroids=af.cluster_centers_
labels = af.labels_
n_clusters_ = len(af.cluster_centers_indices_)

print("类的个数：",n_clusters_)
print("类的中心点：" ,centroids)
print("各点所属的类别：",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

# 画出所有样例点 属于同一分类的绘制同样的颜色
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[af.labels_[i]])
# 画出中心点，用三角形
for i in range(n_clusters_):
    ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

