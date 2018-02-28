#DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from sklearn.cluster import DBSCAN

data=pd.read_csv("prodata.csv")

db = DBSCAN(eps=2, min_samples=10).fit(data) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels))
numSamples = len(data)

print("number of clusters：",n_clusters_)
print("labels of points:",labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y','#6B8E23','#FFA500', '#CD853F','#FFC0CB', '#BC8F8F']

#plot all the points.
for i in range(numSamples):
    ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])
mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y']

ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')
ax.set_title("%d clusters" %n_clusters_)
plt.show()

