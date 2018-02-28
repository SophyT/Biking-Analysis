#mean shift 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth

data=pd.read_csv("prodata.csv")

bandwidth = estimate_bandwidth(data, quantile=0.17)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
numSamples = len(data)
centroids=ms.cluster_centers_
labels = ms.labels_
n_clusters_ = len(np.unique(labels))

print("number of clusters:",n_clusters_)
print("center of clusters:" ,centroids)
print("labels of points：",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

# plot all points.
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])
# plot the center with triangle
for i in range(n_clusters_):
    ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

