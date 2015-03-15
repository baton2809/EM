__author__ = 'artiom'

from sklearn.cluster import KMeans
import numpy as np
from statistics import *

if __name__ == "__main__":
    kmeans = KMeans(n_clusters=2, init='k-means++')
    x = np.loadtxt("covvec", dtype=int)
    kmeans.fit(x.reshape(-1, 1))    # one-dimensional

    # x0, x1 = [], []
    # for i in range(len(x)):
    #     if not kmeans.predict([[x[i]]]):
    #         x0.append(x[i])
    #     else:
    #         x1.append(x[i])
    # lambda0 = mean(x0), lambda1 = mean(x1)

    clusters = kmeans.cluster_centers_
    sum_clusters = int(clusters[0]) + int(clusters[1])
    pi = np.sort(np.array([int(clusters[0])/sum_clusters, int(clusters[1])/sum_clusters]))
    print("lambda : \n{}\npi : \n{}".format(kmeans.cluster_centers_, pi))