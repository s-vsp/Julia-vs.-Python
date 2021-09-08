import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean


def k_means_cost_function(X, centroids, clusters):
    
    J = 0
    for z in range(len(centroids)):
        points_from_cluster = X[np.where(clusters==z)]
        J = J + np.sum([euclidean(points_from_cluster[y], centroids[z]) for y in range(len(points_from_cluster))])
    J = 1/len(X) * J
    
    return J


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
        
        J = k_means_cost_function(X, centroids, clusters)
        print("===============================================")
        print("Loss: {}".format(J))
        print("===============================================")

    return centroids, clusters, J



if __name__ == "__main__":
    
    X, y = make_blobs(n_samples=500, centers=4, n_features=2, random_state=10)
    X = X + 1.2*np.random.normal(0, 1, X.shape)

    plt.figure(dpi=250)
    plt.scatter(X[:,0], X[:,1], color="navy")
    plt.grid(True)
    plt.show()

    cents, clusts, J = k_means_clustering(X,4,5)
    
    plt.figure(dpi=250)
    plt.scatter(X[:,0], X[:,1], color="navy")
    for centroid in range(len(cents)):
        plt.scatter(cents[centroid,0], cents[centroid,1], color="red")
    plt.grid(True)
    plt.show()

    plt.figure(dpi=250)
    for centroid, cluster, color in zip(range(len(cents)), range(len(cents)), ["green", "yellow", "blue", "brown"]):
        plt.scatter(X[:,0][np.where(clusts==cluster)], X[:,1][np.where(clusts==cluster)], color=color, label=f"cluster {cluster}")
        plt.scatter(cents[centroid,0], cents[centroid,1], color="red", edgecolor="black", marker="o")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


    Ks = []
    Js = []
    for n_k in range(1,8):
        cents, clusts, J = k_means_clustering(X,n_k,10)
        Ks.append(n_k)
        Js.append(J)

    plt.figure(dpi=250)
    plt.plot(Ks, Js, color="indigo")
    plt.grid(True)
    plt.title("Elbow method for selecting number of clusters")
    plt.show()
