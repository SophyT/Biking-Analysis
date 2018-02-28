#Affinity Propagation 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import AffinityPropagation

data=pd.read_csv("prodata.csv")

# Affinity Propagation
af = AffinityPropagation(damping=0.87).fit(data)
numSamples = len(data)
centroids=af.cluster_centers_
labels = af.labels_
n_clusters_ = len(af.cluster_centers_indices_)

print("number of clusters:",n_clusters_)
print("centers of clusters:" ,centroids)
print("labels of clusters：",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

# plot all points
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[af.labels_[i]])
# plot the centers with triangle
for i in range(n_clusters_):
    ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

