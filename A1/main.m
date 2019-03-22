% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% Assignment 01 dated March 19 2019
% Author: Harsha HN harshahn@kth.se

%% Exercise 1
close all; clear all; clc;

k = 10;%class 
d = 32*32*3; %image size
N = 10000; %Num of image

A = load('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
I = reshape(A.data', 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
montage(I(:, :, :, 1:500), 'Size', [5,5]);

%% 1. Read in and store the training, validation and test data.

[X, Y, y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
% X: 3072x10,000, Y: 10x10,000, y: 1x10,000

%% 2. Initialize the parameters of the model W and b
W = zeros(k,d); b = zeros(k,1);

for i=1:10
    W(i,:) = 0.01.*randn(1,3072);
end
b = 0.01.*randn(10,1); %W = 0.01.*randn(10,3072);
% W: 10x3072, b: 10x1

%% 3. Check the function runs on a subset of the training data given a random initialization of the network's parameters
n = 2; %Num of images
trainX = X(:, 1:n); trainY = Y(:, 1:n); trainy = y(1:n);
P = EvaluateClassifier(trainX, W, b); %KxN

%% 4. Compute the cost
lambda = 0;%0.01;
J = ComputeCost(trainX, trainY, W, b, lambda);

%% 5 Compute the accuracy
A = ComputeAccuracy(trainX, trainy, W, b);

%% 6. Compute the gradient
[grad_W, grad_b] = ComputeGradients(trainX, trainY, P, W, lambda);

%[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-6);
%[ngrad b, ngrad W] = ComputeGradsNumSlow(trainX(1:20, 1), trainY(:, 1),
%W(:, 1:20), b, lambda, 1e-6);
%% 7. 

%% 8 
