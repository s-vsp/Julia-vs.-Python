using Pkg;
using ScikitLearn;
using PyCall;
using PyPlot;
using Distributions;
using Random;
using CSV;
using DataFrames;

Random.seed!(110)

data = CSV.read("C:\\Users\\Kamil\\My_repo\\OTHER-WAY\\blobs.csv", DataFrame);
df = Matrix(data);
X = df[:, 1:2];
y = df[:, 3];

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


cents, clusts = k_means_clustering(X, 4, 5)


figure()
scatter(X[:,1], X[:,2], color="navy")
for centroid=1:size(cents)[1]
    scatter(cents[centroid,1], cents[centroid,2], color="red")
end


figure()
title("Clustering")
for (centroid, cluster, color) in zip(1:size(cents)[1], 1:size(cents)[1], ["thistle", "black", "gold", "lightblue"])
    scatter(X[findall(==(cluster), clusts),1], X[findall(==(cluster), clusts),2], color=color)
    scatter(cents[centroid,1], cents[centroid,2], color="red", marker="x")
end