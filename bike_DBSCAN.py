#DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import DBSCAN

data=pd.read_csv("prodata.csv")

#DBSCAN
db = DBSCAN(eps=2, min_samples=10).fit(data) #在调整参数中多次测试并观察得聚类结果基本稳定在分为两类，eps大约取值在[1,5]时两类，其他一类
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels))
numSamples = len(data)

print("类的个数：",n_clusters_)
print("各点所属的类别：",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

# 画出所有样例点 属于同一分类的绘制同样的颜色
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y']

ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

