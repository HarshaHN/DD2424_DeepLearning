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
lambda = 0.0;%0.01;
for i=1:10
    W(i,:) = 0.01.*randn(1,3072);
end
b = 0.01.*randn(10,1); %W = 0.01.*randn(10,3072);
% W: 10x3072, b: 10x1

%% 3. Check the function runs on a subset of the training data given a random initialization of the network's parameters
n = 100; %Num of images
trainX = X(:, 1:n); trainY = Y(:, 1:n); trainy = y(1:n);
P = EvaluateClassifier(trainX, W, b); %KxN

%% 4. Compute the cost
J = ComputeCost(trainX, trainY, W, b, lambda);

%% 5 Compute the accuracy
A = ComputeAccuracy(trainX, trainy, W, b);

%% 6. Compute the gradient
%Analytical method
[grad_W, grad_b] = ComputeGradients(trainX, trainY, P, W, lambda);

%Numerical method
%tic % [ngrad_b, ngrad_W] = ComputeGradsNum(trainX, trainY, W, b, lambda, 1e-6); 
% [nsgrad_b, nsgrad_W] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-6); %toc

%Relative error
%rerrW = rerr(grad_W, ngrad_W); rerrb = rerr(grad_b, ngrad_b);

%% 7. Mini-batch gradient descent algorithm
close all; clear all; clc;

k = 10; %class 
d = 32*32*3; %image size
N = 10000; %Num of image

%Load the datasets
[X, Y, y] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
[Xv, Yv, yv] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
[Xt, Yt, yt] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');
% X: 3072x10,000, Y: 10x10,000, y: 1x10,000

% Initialization of parameters & hyperparameters
[W, b] = InitParam(k, d); lambda = 0.0; %0.01; % W: 10x3072, b: 10x1
GDparams.n_batch = 100; GDparams.eta = 0.01; GDparams.n_epochs = 40;
J_train = zeros(1, GDparams.n_epochs); J_val = zeros(1, GDparams.n_epochs);

for e = 1:GDparams.n_epochs
    
    %Random shuffle
    rng(400); shuffle = randperm(N);
    trainX = X(:, shuffle); trainY = Y(:, shuffle);

    %Batches
    ord = 1:N/GDparams.n_batch; %ord = randperm(N/GDparams.n_batch);
    for j=1:max(ord) 
        j_start = (ord(j)-1)*GDparams.n_batch + 1;
        j_end = ord(j)*GDparams.n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, j_start:j_end);
        Ybatch = trainY(:, j_start:j_end);
        [W, b] = MiniBatchGD(Xbatch, Ybatch, GDparams, W, b, lambda);
    end
    
    %Evaluate
    J_train(e) = ComputeCost(X, Y, W, b, lambda);
    J_val(e) = ComputeCost(Xv, Yv, W, b, lambda);
end

%Plot of cost on training & validation set
figure(1); xlim([0 e]);  ylim([min(min(J_train), min(J_val)) max(max(J_train), max(max(J_val)))]);
title('Loss'); xlabel('Epoch'); ylabel('Loss')
plot(J_train); hold on; plot(J_val); hold off; 
legend({'Training loss','Validation loss'},'Location','northeast');

%Accuracy on test set
A = ComputeAccuracy(Xt, yt, W, b)*100;

%Class templates
s_im{10} = zeros(32,32,3);
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i}= (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [1,10]);

%% Exercise 02

