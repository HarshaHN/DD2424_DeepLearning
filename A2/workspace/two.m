% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 02 Assignment dated March 27 2019 
% Author: Harsha HN harshahn@kth.se
% Exercise 1,2,3 : Two layer network

function two
    close all; clear all; clc;

    k = 10; %class 
    m = 50; %Nodes in hidden layer
    d = 32*32*3; %image size
    N = 10000; %Num of training samples
    V = 1000; %Num of validation set

    %Load the datasets
    [Xt, ~, yt] = LoadBatch('../Datasets/cifar-10-batches-mat/test_batch.mat');
    %[Xv, Yv, yv] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_2.mat');
    [X, Y, y] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_1.mat');
%
    [X2, Y2, y2] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_2.mat');
    [X3, Y3, y3] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_3.mat');
    [X4, Y4, y4] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_4.mat');
    [X5, Y5, y5] = LoadBatch('../Datasets/cifar-10-batches-mat/data_batch_5.mat');
    Xv = X5(:,V+1:N); Yv = Y5(:,V+1:N); yv = y5(:,V+1:N);
    %X5 = X5(:, 1:V); Yv = Y5(:, 1:V); yv = y5(:, 1:V);
    X = [X, X2, X3, X4]; Y = [Y, Y2, Y3, Y4];
%}
    N = size(X, 2); %Num of training samples %X: 3072x10,000, Y: 10x10,000, y: 1x10,000

    %Init of hyperparameters
    etaMin = 1e-5; etaMax = 1e-1; t = 0; cycle = 2; %Cyclic learning rate
    v = 2; GDparams.n_batch = 100; batches = N/GDparams.n_batch;
    ns = v*floor(N/GDparams.n_batch); updates = cycle*2*ns; 
    GDparams.n_epochs = updates/batches; % lMin=-5; lMax=-1;
    GDparams.eta = 0.01; 
    
    %Check the gradients Ex2
    %{
    f = 100; n = 5; h = 1e-5; lambda = 0;
    [theta{1,1}, theta{2,1}] = InitParam(m, f);
    [theta{1,2}, theta{2,2}] = InitParam(k, m);
    checkX = X(1:f, 1:n); checkY = Y(:, 1:n);
    [FP] = EvalClassfier( checkX, theta);
    [ga] = CompGradients( checkX, checkY, FP, theta, lambda);
    nW{1} = theta{1,1}; nW{2} = theta{1,2};
    nb{1} = theta{2,1}; nb{2} = theta{2,2};
    [gn] = ComputeGradsNum( checkX, checkY, nW, nb, lambda, h);
    relerr.w1 = rerr(ga{1,1}, gn{1,1}); relerr.w2 = rerr(ga{1,2}, gn{1,2});
    relerr.b1 = rerr(ga{2,1}, gn{2,1}); relerr.b2 = rerr(ga{2,2}, gn{2,2});
    %}

    %Sanity check Ex2 
    %{
    s = 100; N = s; itr =1;
    X = X(:, 1:s); Y = Y(:, 1:s); y = y(:, 1:s);  
    Xv = Xv(:, 1:s); Yv = Yv(:, 1:s); yv = yv(:, 1:s); 
    GDparams.n_epochs = 200; GDparams.eta = 0.01; 
    %}
    
    %Search for lambda
    count = 1;lambda = zeros(1,count);
%
    testA = zeros(1,count); reg = zeros(1,count); vA = zeros(1,count);
    lMin=-4; lMax=-2.5; %Search for lambda
    for rep = 1:count
        l = lMin + (lMax - lMin)*rand(1, 1); lambda(rep) = 10^l;
    end
%}

    for itr = 1:count
        
        % Init of parameters
        theta = {}; tic
        [theta{1,1}, theta{2,1}] = InitParam(m, d);% W: mxd, b: mx1
        [theta{1,2}, theta{2,2}] = InitParam(k, m);% W: kxm, b: kx1

        %lambda(itr) = 0.01; %Ex3 and 4
%{
        %Init of cost 
        J_cap = GDparams.n_epochs; A_cap = J_cap; %updates;
        J_train = zeros(1, J_cap);J_val = zeros(1, J_cap);
        vA = zeros(1, A_cap); tA = zeros(1, A_cap); n = zeros(1, updates);
%}
        %Training
        for e = 1:GDparams.n_epochs %Epochs
            %trainX = X(:, :); trainY = Y(:,:);
            rng(400); shuffle = randperm(N);
            trainX = X(:, shuffle); trainY = Y(:, shuffle);

            %Batchwise parameter updation
            ord = randperm(batches); %Random shuffle of batches
            
            for j=1:batches 
                t = t + 1; % Increment update count
                j_start = (ord(j)-1)*GDparams.n_batch + 1;
                j_end = ord(j)*GDparams.n_batch;
                inds = j_start:j_end;
                Xbatch = trainX(:, inds);
                Ybatch = trainY(:, inds);

                %Updates
                GDparams.eta = cyclic(t, ns, etaMax, etaMin); n(t) = GDparams.eta;
                [theta, ~] = MiniBatchGD(Xbatch, Ybatch, GDparams, theta, lambda(itr));
                
                %Evaluate Loss
                %J_train(t) = ComputeCost(X, Y, theta, lambda);
                %J_val(t) = ComputeCost(Xv, Yv, theta, lambda);
                
                %Evaluate Accuracy
                %tA(t) = ComputeAccuracy(X, y, theta)*100; 
                %vA(t) = ComputeAccuracy(Xv, yv, theta)*100; 
                
                %sprintf('Total updates: %d, Epoch %d - Batch %d', t, e, j)
            end

            %Evaluate losses
            %J_train(e) = ComputeCost(X, Y, theta, lambda);
            %J_val(e) = ComputeCost(Xv, Yv, theta, lambda);

            %Accuracy
            %tA(e) = ComputeAccuracy(X, y, theta)*100; 
            %vA(e) = ComputeAccuracy(Xv, yv, theta)*100; 

            sprintf('Iter %d, Epoch %d', itr, e)
            %sprintf('Epoch %d, total updates %d', e, t)
        end
%{       
        %Plot of cost on training & validation set
        figure(1); plot(J_train); hold on; plot(J_val); hold off; 
        xlim([0 e]); ylim([0 4]); %ylim([min(min(J_train), min(J_val)) max(max(J_train), max(max(J_val)))]);
        title('Cost Plot'); xlabel('Epoch'); ylabel('Cost'); grid on;
        legend({'Training','Validation'},'Location','northeast');
        sprintf('Total number of update steps is %2d', t);
%
        %Plot of Accuracy on training & validation set 
        figure(2); plot(tA); hold on; plot(vA); hold off; 
        xlim([0 e]); ylim([30 75]);
        title('Accuracy Plot'); xlabel('Epoch'); ylabel('Accuracy'); grid on;
        legend({'Training','Validation'},'Location','northeast');
        sprintf('Total number of update steps is %2d', t);
%}        
        %Accuracy on test data
        %testA = ComputeAccuracy(Xt, yt, theta)*100;
        %sprintf('Accuracy on test set is %2.2f %',testA) 
%{
        tA = ComputeAccuracy(X, y, theta)*100; 
        vA = ComputeAccuracy(Xv, yv, theta)*100; 
        sprintf('Accuracy on training set is %2.2f %',A)
        sprintf('Accuracy on validation set is %2.2f %',vA)        
%}         
        %Grid search: Coarse/fine search for optimal lambda
        sprintf('Count: %d completed in %2.2f', itr, toc)
        reg(itr) = log10(lambda(itr));
        vA(itr) = ComputeAccuracy(Xv, yv, theta)*100; vA(itr)

        %Cyclic learning
        figure(1); plot(n)
        title('Cyclic eta'); xlabel('Updates'); ylabel('eta'); grid on;
        
        %sprintf('Total time taken %2.2f', toc)
    end
    figure(2); scatter(reg, vA); grid on;
    title('Accuracy vs Lambda (Fine search)'); xlabel('lambda (logarithmic)'); ylabel('Accuracy');
end

function [eta] = cyclic(t, ns, etaMax, etaMin)
    %Evaluates cyclic learning rate
    
    slope = (etaMax - etaMin)./ns;
    if mod(t, 2*ns) < ns
        %upward movement;
        eta = etaMin + mod(t, 2*ns)*slope;
    else 
        %downward movement;
        eta = etaMax - mod(t, ns)*slope;
    end
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

function [FP] = EvalClassfier(X, theta)
    %2 layer: ReLU and softmax
    %P: KxN and H: mxN
   
    s1 = theta{1,1}*X + theta{2,1}; % mxd*dxN + mx1 = mxN
    H = max(0, s1); % mxN
    s = theta{1,2}*H + theta{2,2}; % Kxm*mxN + Kx1 = KxN
    P = softmax(s); % KxN
    FP = {}; FP{1} = H; FP{2} = P; %Forward Pass layer
end

function [gradTheta] = CompGradients(X, Y, FP, theta, lambda)
    %Compute the gradients through back propagation
    % Y or P: KxN, X: dxN, theta = W: Kxd, b: Kx1, H: mxN
    
    %Initialize
    gW1 =0; gW2 =0; gb1 =0; gb2 =0;
    
    %Update loop
    N = size(X, 2); H = FP{1}; P = FP{2};
    for i = 1:N % ith image
        
        g = -(Y(:,i)-P(:,i))'; %1xK
        gW2 = gW2 + g' * H(:,i)'; %Kx1*1xm = Kxm
        gb2 = gb2 + g'; %Kx1
        
        %mxN
        g = g*theta{1,2}; %1xK*K*m = 1xm
        g = g*diag((H(:,i)>0)); %1xm*mxm = 1xm

        gW1 = gW1 + g'*X(:,i)'; %mxN*Nxd = mxd
        gb1 = gb1 + g'; % mxN
    end

    gradTheta{1,1} = (1./size(X, 2)).* (gW1) + lambda .*2*theta{1,1};
    gradTheta{1,2} = (1./size(X, 2)).* (gW2) + lambda .*2*theta{1,2};
    gradTheta{2,1} = (1./size(X, 2)).* (gb1);
    gradTheta{2,2} = (1./size(X, 2)).* (gb2);    
end

function [thetaStar, FP] = MiniBatchGD(X, Y, GDparams, theta, lambda)
%Mini batch Gradient Descent Algo
    %Predict
    [FP] = EvalClassfier(X, theta);
 
    %Compute gradient
    [gradTheta] = CompGradients(X, Y, FP, theta, lambda);
    
    %Update the parameters in theta
    thetaStar{1,1} = theta{1,1} - GDparams.eta * gradTheta{1,1};
    thetaStar{1,2} = theta{1,2} - GDparams.eta * gradTheta{1,2};
    thetaStar{2,1} = theta{2,1} - GDparams.eta * gradTheta{2,1};
    thetaStar{2,2} = theta{2,2} - GDparams.eta * gradTheta{2,2};
end

function J = ComputeCost(X, Y, theta, lambda)
    % Compute cost
    % Y: KxN, X: dxN, W: Kxd, b: Kx1, lambda
    [FP] = EvalClassfier(X, theta); %KxN
    L = -log(Y' * FP{2}); %NxN
    totalLoss = trace(L); %sum(diag(L))
    R = sumsqr(theta{1,1}) + sumsqr(theta{1,2});
    J = (totalLoss)./ size(X, 2) + lambda.*R;
end

function A = ComputeAccuracy(X, y, theta)
    %Compute the accuracy
    % y: 1xN, X: dxN, W: Kxd, b: Kx1, lambda    
    [FP] = EvalClassfier(X, theta); %KxN
    [~, argmax] = max(FP{2});
    c = (argmax == y);
    A = sum(c)/size(c,2);
end

%------------------------------------------------------------
%Correctness check of the Gradient
function [gn] = ComputeGradsNum(X, Y, W, b, lambda, h)
    %Compute Gradient numerically
    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    [c] = ComputeCost(X, Y, mConv(W,b), lambda);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            [c2] = ComputeCost(X, Y, mConv(W, b_try), lambda);
            grad_b{j}(i) = (c2-c) / h;
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})   
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            [c2] = ComputeCost(X, Y, mConv(W_try, b), lambda);
            grad_W{j}(i) = (c2-c) / h;
        end
    end
    gn = mConv(grad_W, grad_b);
    %gn{1,:} = grad_W; gn{2,:} = grad_b;
end

function [rerr] = rerr(ga, gn)
    %Compute relative error
    rerr = sum(sum(abs(ga - gn)./max(eps, abs(ga) + abs(gn))))./ numel(ga);
end

function [theta] = mConv(W,b)
    %Mathematical convenience
    theta{1,1} = W{1}; theta{1,2} = W{2};
    theta{2,1} = b{1}; theta{2,2} = b{2};
end    

%{
    %Class templates
%     s_im{10} = zeros(32,32,3);
%     nW{1} = theta{1,1}; nW{2} = theta{1,2};
%     nb{1} = theta{2,1}; nb{2} = theta{2,2};
%     for k = 1:2
%          for i=1:10
%              im = reshape(nW{k}(i, :), 32, 32, 3);
%              s_im{i}= (im - min(im(:))) / (max(im(:)) - min(im(:)));
%              s_im{i} = permute(s_im{i}, [2, 1, 3]);
%          end
%          figure(); title('Class Template images'); 
%          montage(s_im, 'Size', [1,10]);
%     end 
%}
