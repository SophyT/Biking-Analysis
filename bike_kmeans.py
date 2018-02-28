#kmeans算法须人为指定类型数目,受初始点影响显著，更适用于各类数量相似的数据集，但运行速度最快，使用最广
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D

data=pd.read_csv("prodata.csv")

# 假设可分为2-7类，进行遍历
for k in range(2, 8):
    clf = KMeans(n_clusters=k,init='k-means++')
    s = clf.fit(data)
    numSamples = len(data)
    labels = clf.labels_
    centroids=clf.cluster_centers_
    print("类的个数：",k)
    print("类的中心点：" ,centroids)
    print("各点所属的类别：",labels)
    print("聚类结果衡量参数",clf.inertia_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mark = ['c', 'b', 'g', 'k', 'r', 'm', 'y']

    # 画出所有数据点，属于同一分类的绘制同样的颜色
    for i in range(numSamples):
        ax.scatter(data.iloc[i,0], data.iloc[i,1],data.iloc[i,2],c=mark[labels[i]])

    # 画出中心点，三角形表示
    for i in range(k):
        ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2],c=mark[i],marker='v')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("%d clusters" %k)
    plt.show()






