% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 03 Assignment dated April 17 2019 
% Author: Harsha HN harshahn@kth.se
% Character-level Convolutional Networks for Text Classification

function three
    close all; clear; clc; 
    TotalTime = tic;

    %0.0 Data preprocessing
    ExtractNames; L = load('assignment3_names.mat');
    C = unique(cell2mat(L.all_names));%List of uniq characters
    d = numel(C); %Num of uniq characters
    K = length(unique(L.ys)); %Number of unique classes
    N = length(L.ys);% Num of names in the dataset
    n_len = 0; %Maximum length of names
    for i = 1:N
        n_len = max(length(L.all_names{i}), n_len);
    end

    % Character to Index mapping
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');
    for i = 1: length(C)
        char_to_ind(C(i)) = i;
        ind_to_char(i) = C(i);
    end

    % Input (names) encoding into dxn_len
    X = zeros(d, n_len, N);
    for i = 1:N
        temp = L.all_names{i};
        n_name = length(temp);
        for j = 1: n_name
            ind = char_to_ind(temp(j));
            X(:, j, i) = bsxfun(@eq, 1:d, ind)'; %one-hot encoding
        end
    end

    %% 0.1 Partition into Train data and Validation data
    data_fname = 'Validation_Inds.txt';
    fid = fopen(data_fname,'r'); S = fscanf(fid,'%c');
    fclose(fid); names = strsplit(S, ' ');

    valInd = str2double(names)';
    trainInd = setdiff(1:N,valInd)';trainN = size(trainInd, 1);
    valX = X(:, :, valInd); valY= ys(valInd); %Validation set
    trainX = X(:, :, trainInd); trainY = ys(trainInd); %Training set

    %% 0.2 Set hyperparameters
    n1 = 5; %Num of filters at layer 1
    k1 = 5; %width of filter at layer 1
    n2 = 5; %Num of filters at layer 2
    k2 = 5; %width of filter at layer 2
    n_len1 = n_len - k1 + 1; n_len2 = n_len1 - k2 + 1; widW = n2 * n_len2;
    
    %Weight initialization
    sig1 = sqrt(2/(1*d*n_len)); sig2 = sqrt(2/(n1*n_len1)); sig3 = sqrt(2/widW);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(K, widW)*sig3;

    %GD Params
    GDparams.eta = 0.01; GDparams.rho = 0.9; t = 0;
    GDparams.n_epochs = 15; GDparams.n_batch = 38; %2, 19, 38, 521
    batches = trainN/GDparams.n_batch;
    
    %% 0.3 Construct the convolution matrices
   
%{
    %Sanity check
    check = load('DebugInfo.mat');
    x_input = check.X_input;
    
    %Convolution into matrix multiplication
    checkMF = MakeMFMatrix(check.F, n_len);
    checkMX = MakeMXMatrix(x_input, 28, 5, 4);
        
    s1 = checkMX * check.vecF; 
    s2 = checkMF * x_input(:);    
    equality = isequal(s1, s2, check.vecS)
%}
    
    %Initialization
    J_train = zeros(GDparams.n_epochs, 1); J_val = zeros(GDparams.n_epochs, 1); %Cost
    tA = zeros(GDparams.n_epochs, 1); vA = zeros(GDparams.n_epochs, 1); %Accuracy
    
    %Pre-computes
    VX = MakeVecXMatrix(trainX, d, k1);
    
    %Training
    for e = 1:GDparams.n_epochs %Epochs
        EpochTime = tic;
        %Random shuffle
        rng(400); shuffle = randperm(trainN);
        trainX = trainX(:, :, shuffle); VX = VX(:, :, shuffle); trainY = trainY(shuffle);

        %Batchwise parameter updation
        ord = randperm(batches); %Random shuffle of batches

        for j=1:batches 
            t = t + 1; % Increment update count
            j_start = (ord(j)-1)*GDparams.n_batch + 1;
            j_end = ord(j)*GDparams.n_batch;
            inds = j_start:j_end;
            Xbatch = trainX(:, :, inds); VXbatch = VX(:, :, inds); 
            Ybatch = trainY(inds);

            %Updates
            %GDparams.eta = cyclic(t, ns, etaMax, etaMin); n(t) = GDparams.eta;
            [ConvNet, ~] = MiniBatchGD(Xbatch, VXbatch, Ybatch, ConvNet, GDparams);
%{
            %Evaluate Loss
            %J_train(t) = ComputeCost(X, Y, theta, lambda);
            %J_val(t) = ComputeCost(Xv, Yv, theta, lambda);

            %Evaluate Accuracy
            %tA(t) = ComputeAccuracy(X, y, theta)*100; 
            %vA(t) = ComputeAccuracy(Xv, yv, theta)*100; 
%}
            %sprintf('Total updates: %d, Epoch %d - Batch %d', t, e, j)
        end
        
        %Evaluate losses
        [J_train(e), YpredT] = ComputeLoss(trainX, trainY, ConvNet);
        [J_val(e), YpredV] = ComputeLoss(valX, valY, ConvNet);
%{        
        %J_train(e) = 0; bN = 9800;
        %for set = 1:tset %Mathematical convenience
        %    str = (set-1)*bN + 1; en = set*bN;
        %    J_train(e) =  J_train(e) + ComputeCost(X(:, str:en), Y(:, str:en), theta, lambda);
        %    lambda(itr) = 0;
        %end
        %lambda(itr) = 10^-2.98; J_train(e) = J_train(e)/tset;
        %J_val(e) = ComputeCost(Xv, Yv, theta, lambda);
%}
        %Accuracy
        tA(e) = ComputeAccuracy(trainY, YpredT)*100; 
        vA(e) = ComputeAccuracy(valY, YpredV)*100; 

        %sprintf('Iter %d, Epoch %d', itr, e)
        sprintf('Epoch %d in %d, Total updates %d', e, toc(EpochTime), t)
    end
%      
        %Plot of cost on training & validation set
        figure(1); plot(J_train); hold on; plot(J_val); hold off; 
        xlim([1 e]); ylim([min(min(J_train), min(J_val)) max(max(J_train), max(max(J_val)))]);
        title('Loss Plot'); xlabel('Epoch'); ylabel('Loss'); grid on;
        legend({'Training','Validation'},'Location','northeast');
        sprintf('Total number of update steps is %2d', t);
%
        %Plot of Accuracy on training & validation set 
        figure(2); plot(tA); hold on; plot(vA); hold off; 
        xlim([1 e]); ylim([30 100]);
        title('Accuracy Plot'); xlabel('Epoch'); ylabel('Accuracy'); grid on;
        legend({'Training','Validation'},'Location','northeast');
        sprintf('Total number of update steps is %2d', t);
%} 
    toc(TotalTime)
end

function A = ComputeAccuracy(Ybatch, Ypred)
    %Compute the accuracy
    c = zeros(size(Ypred));
    c = (Ybatch == Ypred);
    A = sum(c)/size(c,1);
end

function [J, Ypred] = ComputeLoss(Xbatch, Ybatch, ConvNet)
    %Compute Loss
    %X_batch: n_len*d x n
    [Ypred, FP, ~] = ForwardPass(Xbatch, ConvNet);
    
    Loss = 0; samples = size(Xbatch,3);
    for i=1:samples
        p = FP{3,i};
        Loss = Loss - log(p(Ybatch(i))) ;
    end
    J = Loss ./ samples;
end

function [ConvNetStar, FP] = MiniBatchGD(Xbatch, VXbatch, Ybatch, ConvNet, GDparams)
%Mini batch Gradient Descent Algo
    %Predict
    [~, FP, MF] = ForwardPass(Xbatch, ConvNet);
 
    %Compute gradient
    [gradConvNet] = CompGradients(VXbatch, Ybatch, ConvNet, FP, MF);
    
    %Update the parameters in theta
    ConvNetStar.F{1} = ConvNet.F{1} - GDparams.eta * gradConvNet.F{1};
    ConvNetStar.F{2} = ConvNet.F{2} - GDparams.eta * gradConvNet.F{2};
    ConvNetStar.W = ConvNet.W - GDparams.eta * gradConvNet.W;
end

function [Yp, FP, MF] = ForwardPass(Xbatch, ConvNet)
    %2 layer CNN: Forward Pass
    %YP: samples, FP: 3xsamples of X, MF: 1x2 of F1, F2
    %Xbatch: d x n_len x samples, ConvNet: F1, F2, W
    
    [~, n_len, samples] = size(Xbatch); %[k, ~] = size(ConvNet.W);
    [~, k1, n1] = size(ConvNet.F{1}); [~, k2, n2] = size(ConvNet.F{2});
    n_len1 = n_len - k1 + 1;  n_len2 = n_len1 - k2 + 1;
    
    MF{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
    MF{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
    
    FP = cell(3,samples); Yp = zeros(samples, 1);
    for i = 1:samples
        in = Xbatch(:,:,i);
        FP{1,i} = reshape(max(MF{1} * in(:), 0), [n1, n_len1]); %ReLU
        FP{2,i} = reshape(max(MF{2} * FP{1,i}(:), 0), [n2, n_len2]); %ReLU
        S = ConvNet.W * FP{2,i}(:);  FP{3,i} = softmax(S); %Softmax
        [~, Yp(i)] = max(FP{3,i}); %Predict 
    end
end

function [gradConvNet] = CompGradients(VXbatch, Ybatch, ConvNet, FP, MF)
    %Compute gradients via back propagation w.r.t  vec(F1), vec(F2), W
    %Xbatch: d x n_len x samples, Ybatch: samples, ConvNet: F1, F2, W
    %FP: 3xsamples of X, MF: 1x2 of F1, F2
    
    [k, w] = size(ConvNet.W); LW = zeros([k, w]);
    [n1, k2, n2] = size(ConvNet.F{2}); LF2 = zeros([n1, k2, n2]);
    [d, k1, n1] = size(ConvNet.F{1}); LF1 = zeros([d, k1, n1]);
    samples = size(Ybatch,1); n_len2 = w./n2;
    %[~, n_len, samples] = size(Xbatch); %n_len1 = n_len - k1 + 1;
    
    for i = 1:samples
        Yhot = bsxfun(@eq, 1:k, Ybatch(i))';
        %Softmax
        G3 = -(Yhot - FP{3, i});
        %Layer 3
        LW = LW + G3 * FP{2,i}(:)'; %W*X: 18x1 X 1x55 = 18x55
        
        %Layer 2
        G2 = ConvNet.W' * G3; %W*X: 55x18 X 18x1 = 55x1
        G2 = G2 .* (FP{2,i}(:)>0); %ReLU:  55x1 .* 55x1 = 55x1

        VX = MakeVecXMatrix(FP{1,i}, n1, k2); %11x25
        Gp = reshape(G2, [n2, n_len2]); %5x11
        v1 = VX' * Gp'; %11x25' X 5x11' = 25x5
        %MX = MakeMXMatrix(FP{1,i}, n1, k2, n2); %55x125
        %v1_check = G2' * MX; isequal(v1(:), v1_check');
        
        LF2 = LF2 + reshape(v1, size(LF2));
        %Layer 1
        G1 = MF{2}' * G2; %W*X: 75x55 X 55x1 = 75x1
        G1 = G1 .* (FP{1,i}(:)>0); %ReLU:  75x1 .* 75x1 = 75x1
        %MX{2} = MakeMXMatrix(Xbatch(:,:,i), d, k1, n1);
        MXpre = VXtoMX(VXbatch(:,:,i), n1);
        v2 = G1' * MXpre;        
        LF1 = LF1 + reshape(v2, size(LF1));
    end
    gradConvNet.W = (1/samples)*LW; 
    gradConvNet.F{1} = (1/samples)*LF1; 
    gradConvNet.F{2} = (1/samples)*LF2;
end

function VX = MakeVecXMatrix(trainX, d, k)
    [~, n_len, samples] = size(trainX);
    VX = zeros(n_len-k+1, d*k, samples);
    
    for j = 1:samples
        for i = 1: n_len-k+1
            temp = trainX(:, i:k+i-1, j);
            VX(i,:,j) = temp(:)';
        end
    end
end

function MX = VXtoMX(VX, nf)
    MX = zeros(size(VX)*nf);
    s = 1; e = nf;
    for i=1:size(VX,1)
        MX(s:e, :) = kron(eye(nf), VX(i,:));
        %MX(s:e, :) = MX(s:e, :) + circshift([VF, zeros(nf, (n_len-k)*dd)], [0 shift]) VX(i,:);
        s = s + nf; e = e + nf;
    end
end

function MX = MakeMXMatrix(x_in, d, k, nf)
    %MX: (n_len-k+1)*nf X k*nf*d
    [~, n_len] = size(x_in);
    MX = zeros((n_len-k+1)*nf, k*nf*d);
    VX = zeros(n_len-k+1, d*k);
    
    s = 1; e = nf;
    for i=1:(n_len-k+1)
        temp = x_in(:, i:k+i-1);
        VX(i, :) = temp(:)';
        MX(s:e, :) = kron(eye(nf), VX(i, :));
        s = s + nf; e = e + nf;
    end
end

function MF = MakeMFMatrix(F, n_len)
    % MF: (nlen-k+1)*nf X nlen*dd
    [dd, k, nf] = size(F);
    MF = zeros((n_len-k+1)*nf, n_len*dd);
    %VF update
    VF = zeros(nf, dd*k);
    for i = 1:nf
        temp = F(:, :, i);
        VF(i, :) = temp(:)'; %nf X dd*k
    end
    %MF update
    s = 1; e = nf;
    for j = 1:(n_len-k+1)
        shift = (j-1)*dd;
        MF(s:e, :) = MF(s:e, :) + circshift([VF, zeros(nf, (n_len-k)*dd)], [0 shift]);
        s = s + nf; e = e + nf;
    end
end
