#kmeans
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D

data=pd.read_csv("prodata.csv")

# Suppose number of clusters is between 2 and 7
for k in range(2, 8):
    clf = KMeans(n_clusters=k,init='k-means++')
    s = clf.fit(data)
    numSamples = len(data)
    labels = clf.labels_
    centroids=clf.cluster_centers_
    print("number of clusters",k)
    print("center of clustrs：" ,centroids)
    print("labels of points：",labels)
    print("inertia",clf.inertia_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y']

    # Plot all the point. Use same the same color for the points in the same cluster.
    for i in range(numSamples):
        ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])

    # Plot the centers with triangle.
    for i in range(k):
        ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("%d clusters" %k)
    plt.show()






