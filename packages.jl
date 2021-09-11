using Pkg;
using ScikitLearn;
using PyCall;
using PyPlot;
using Distributions;
using Random;
using CSV;
using DataFrames;
using ScikitLearn;

Random.seed!(110)

data = CSV.read("C:\\Users\\Kamil\\My_repo\\OTHER-WAY\\blobs.csv", DataFrame);
df = Matrix(data);
X = df[:, 1:2];
y = df[:, 3];
