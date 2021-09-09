using Pkg;
using ScikitLearn;
using PyCall;
using PyPlot;
using Distributions;
using Random;

Random.seed!(110)

@sk_import datasets: make_blobs

X, y = make_blobs(n_samples=500, centers=4, n_features=2);
X = X + 1.2*rand(Normal(), size(X));

# Starting figure
figure()
scatter(X[:,1], X[:,2], color="navy")


function k_means_clustering(X, K, n_iters)

    global centroids_ids = [];

    for n_k=1:K
        append!(centroids_ids, rand(1:size(X)[1], 1));
    end

    global centroids = X[centroids_ids,:];

    for iter=1:n_iters

        # Cluster assignment step
        global clusters = zeros(size(X)[1]);
        for i=1:size(X)[1]
            distances = []
            for d=1:K
                distance = sqrt.(sum((X[i,:] - centroids[d,:]).^2));
                #distance = euclidean(X[i,:], centroids[d,:])
                append!(distances, distance);
            end
            clusters[i] = argmin(distances)
        end

        # Move centorids step
        for k=1:K
            centroids[k,:] = (1/length(clusters[findall(==(k), clusters)])) * sum(X[findall(==(k), clusters), :], dims=1)
        end
    end
    return centroids, clusters
end


cents, cluts = k_means_clustering(X, 4, 5)

figure()
scatter(X[:,1], X[:,2], color="navy")
for centroid=1:size(cents)[1]
    scatter(cents[centroid,1], cents[centroid,2], color="red")
end