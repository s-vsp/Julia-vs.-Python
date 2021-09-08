using Pkg;
using ScikitLearn;
using PyCall;
using PyPlot;
using Distributions;
using Random;

Random.seed!(110)

@sk_import datasets: make_blobs


X, y = make_blobs(n_samples=500, centers=4, n_features=2)
X = X + 1.2*rand(Normal(), size(X))

figure()
scatter(X[:,1], X[:,2], color="navy")