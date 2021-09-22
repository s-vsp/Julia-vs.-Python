import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean


def k_means_clustering(X, K, n_iters):
    
    centroids_inds = []
    
    for n_k in range(K):
        ind = np.random.randint(0,len(X),1)
        centroids_inds.extend(ind)
    
    centroids = X[centroids_inds]
    
    for iteration in range(n_iters):

        # Cluster assignment step
        clusters = np.zeros((len(X)))
        for i in range(len(X)):
            distances = []
            for d in range(K):
                distance = euclidean(X[i], centroids[d])
                distances.append(distance)
            clusters[i] = distances.index(min(distances))
        
        # Move centroids step
        for k in range(K):
            centroids[k] = 1/len(clusters[clusters==k]) * np.sum(X[clusters==k], axis=0)

    return centroids, clusters



if __name__ == "__main__":

    data = pd.read_csv("C:\\Users\\Kamil\\My_repo\\OTHER-WAY\\blobs.csv").to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    plt.figure(dpi=250)
    plt.title("Raw data")
    plt.scatter(X[:,0], X[:,1], color="navy")
    plt.show()

    cents, clusts = k_means_clustering(X,4,5)

    plt.figure(dpi=250)
    plt.title("Centroids")
    plt.scatter(X[:,0], X[:,1], color="navy")
    for centroid in range(len(cents)):
        plt.scatter(cents[centroid,0], cents[centroid,1], color="red", marker="x")
    plt.show()

    plt.figure(dpi=250)
    plt.title("Clustering")
    for centroid, cluster, color in zip(range(len(cents)), range(len(cents)), ["thistle", "black", "gold", "lightblue"]):
        plt.scatter(X[:,0][clusts==cluster], X[:,1][clusts==cluster], color=color, 
                    label="cluster {}".format(cluster))
        plt.scatter(cents[centroid,0], cents[centroid,1], color="red", marker="x")
    plt.show()
