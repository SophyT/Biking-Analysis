#mean shift 不需要制定类的数量，受初始中心点和数据影响小，稳定性高于kmeans，但在大数据集上运行速度慢
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth

data=pd.read_csv("prodata.csv")

bandwidth = estimate_bandwidth(data, quantile=0.17)#可调参数：quantile [0,1],quantile值越大，类别数越少。在该小数据集上，经测试得，当取[0.6,1]时分为两类，当取[0.3,0.6]时分为三类，当取[0.17,0.3]时分为四类。分类结果相对稳定。
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
numSamples = len(data)
centroids=ms.cluster_centers_
labels = ms.labels_
n_clusters_ = len(np.unique(labels))

print("类的个数：",n_clusters_)
print("类的中心点：" ,centroids)
print("各点所属的类别：",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

# 画出所有样例点 属于同一分类的绘制同样的颜色
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])
# 画出中心点，用三角形
for i in range(n_clusters_):
    ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

