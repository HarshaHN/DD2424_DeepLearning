% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 02 Assignment dated March 27 2019 
% Author: Harsha HN harshahn@kth.se
% Two layer network
% Exercise 1

function one
    close all; clear all; clc;

    k = 10; %class 
    m = 50; %Nodes in hidden layer
    d = 32*32*3; %image size
    N = 10000; %Num of training samples

    %Load the datasets
    [X, Y, ~] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_1.mat');
    [Xv, Yv, ~] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_2.mat');
    [Xt, ~, yt] = LoadBatch('../Datasets/cifar-10-batches-mat/test_batch.mat');
    % X: 3072x10,000, Y: 10x10,000, y: 1x10,000

    % Init of parameters
    %theta = {};
    [theta.W1, theta.b1] = InitParam(m, d);% W: mxd, b: mx1
    [theta.W2, theta.b2] = InitParam(k, m);% W: kxm, b: kx1
    
    %Init of hyperparameters
    lambda = 0; GDparams.eta = 0.01; 
    GDparams.n_batch = 50; GDparams.n_epochs = 1;
    
    %check = ComputeCost(X, Y, theta, lambda); %remove
    %Init of cost 
    J_train = zeros(1, GDparams.n_epochs); 
    J_val = zeros(1, GDparams.n_epochs);
    
    %Training
    for e = 1:GDparams.n_epochs %Epochs

        %Random shuffle
        N = GDparams.n_batch; %remove
        rng(400); shuffle = randperm(N);
        trainX = X(:, shuffle); trainY = Y(:, shuffle);

        %Batchwise parameter updation
        ord = 1:N/GDparams.n_batch; %ord = randperm(N/GDparams.n_batch);
        for j=1:max(ord) 
            j_start = (ord(j)-1)*GDparams.n_batch + 1;
            j_end = ord(j)*GDparams.n_batch;
            inds = j_start:j_end;
            Xbatch = trainX(:, inds);
            Ybatch = trainY(:, inds);
            [ theta] = MiniBatchGD(Xbatch, Ybatch, GDparams, theta, lambda);
        end

        %Evaluate losses
        J_train(e) = ComputeCost(X, Y, theta, lambda);
        J_val(e) = ComputeCost(Xv, Yv, theta, lambda);
    end

    %Plot of cost on training & validation set
    figure(1); plot(J_train); hold on; plot(J_val); hold off; 
    xlim([0 e]); ylim([0 20]); %ylim([min(min(J_train), min(J_val)) max(max(J_train), max(max(J_val)))]);
    title('Total loss'); xlabel('Epoch'); ylabel('Loss'); grid on;
    legend({'Training loss','Validation loss'},'Location','northeast');

    %Accuracy on test set
    A = ComputeAccuracy(Xt(:,1:10), yt(:,1:10), theta)*100; 
    sprintf('Accuracy on test data is %2.2f %',A)

    %Class templates
    s_im{10} = zeros(32,32,3);
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i}= (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    figure(2); title('Class Template images'); montage(s_im, 'Size', [1,10]);
end

function [X, Y, y] = LoadBatch(filename)
    %Load .mat files into workspace
    
    A = load(filename); 
    X = batchNorm(double(A.data')./255);% dxN 3072x10,000  
    Y = bsxfun(@eq, 1:10, A.labels+1)';% KxN 10x10,000
    y = (A.labels + 1)';% 1xN 1x10,000
end

function [normX] = batchNorm(X)
    %Batch Normalization
    meanX = mean(X, 2);
    stdX = std(X, 0, 2);len = size(X, 2);
    zcX = X - repmat(meanX, [1, len]);
    normX = zcX ./ repmat(stdX, [1, len]);
end

function [W, b] = InitParam(r, c)
    %Initialisation of model parameters
    % W: rxc, b: rx1
    
    W = zeros(r,c); b = zeros(r,1);
    rng(400);
    for i=1:r
        W(i,:) = 1/sqrt(c).*randn(1,c);
    end
    %b = 0.01.*randn(r,1); %W = 1/sqrt(c).*randn(r,c);
end

function P = EvalClassfier(X, theta)
    %P: KxN 2 layer: ReLU and softmax
    %theta = W1: mxd, b1: mx1  W2: kxm, b2: kx1
   
    s1 = theta.W1*X + theta.b1; % mxd*dxN + mx1 = mxN
    h = max(0, s1); % mxN
    s = theta.W2*h + theta.b2; % Kxm*mxN + Kx1 = KxN
    P = softmax(s); % KxN
end

function [ gradTheta] = CompGradients(X, Y, P, theta, lambda)
    %Compute the gradients through back propagation
    % Y or P: KxN, X: dxN, theta = W: Kxd, b: Kx1
    
    %Initialize
    LossW = 0; Lossb = 0;
    
    %Update loop
    N = size(X, 2);
    for i = 1:N % ith image
        g = -(Y(:,i)-P(:,i))'; %NxK
        LossW = LossW + g' * X(:,i)';
        Lossb = Lossb + g';
    end
    M = 0; %Momentum
    gradTheta.W1 = (1./size(X, 2)) * (LossW) + lambda .*2*theta.W1;
    gradTheta.W2 = (1./size(X, 2)) * (LossW) + lambda .*2*theta.W2;
    gradTheta.b1 = (1./size(X, 2)) * (Lossb);
    gradTheta.b2 = (1./size(X, 2)) * (Lossb);    
end

function [ thetaStar] = MiniBatchGD(X, Y, GDparams, theta, lambda)
    
    %Predict
    P = EvalClassfier(X, theta);
 
    %Compute gradient
    [ gradTheta] = CompGradients(X, Y, P, theta, lambda);
    
    %Update the parameters in theta
    thetaStar.W1 = theta.W1 - GDparams.eta * gradTheta.W1;
    thetaStar.W2 = theta.W2 - GDparams.eta * gradTheta.W2;
    thetaStar.b1 = theta.b1 - GDparams.eta * gradTheta.b1;
    thetaStar.b2 = theta.b2 - GDparams.eta * gradTheta.b2;
end

function J = ComputeCost(X, Y, theta, lambda)
    % Compute cost
    % Y: KxN, X: dxN, W: Kxd, b: Kx1, lambda

    P = EvalClassfier(X, theta); %KxN
    L = -log(Y' * P); %NxN
    totalLoss = trace(L); %sum(diag(L))
    R = sumsqr(theta.W1) + sumsqr(theta.W2);
    J = (totalLoss)./ size(X, 2) + lambda.*R;
end

function A = ComputeAccuracy(X, y, theta)
    % y: 1xN, X: dxN, W: Kxd, b: Kx1, lambda    
    P = EvalClassfier(X, theta); %KxN
    [~, argmax] = max(P);
    c = (argmax == y);
    A = sum(c)/size(c,2);
end

function [rerr] = rerr(ga, gn)
    %Compute relative error
    rerr = sum(sum(abs(ga - gn)./max(eps, abs(ga) + abs(gn))))./ numel(ga);
end

%------------------------------------------------------------
function P = EvalClassifier01(X, W, b)
    %Linear equation and softmax func
    s = W*X + b; % Kxd*dxN + Kx1 = KxN
    P = softmax(s); % KxN

end
