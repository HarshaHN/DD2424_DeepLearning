% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 03 Assignment dated April 17 2019 
% Author: Harsha HN harshahn@kth.se
% Character-level Convolutional Networks for Text Classification

function three
    close all; clear; clc; 
    TotalTime = tic;

    %% 0.0 Data preprocessing
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
    %File ops
    data_fname = 'Validation_Inds.txt';
    fid = fopen(data_fname,'r'); S = fscanf(fid,'%c');
    fclose(fid); names = strsplit(S, ' ');
    
    %Partition based on indices
    valInd = str2double(names)';
    trainInd = setdiff(1:N,valInd)'; trainN = size(trainInd, 1);
    valX = X(:, :, valInd); valY= ys(valInd); %Validation set
    trainX = X(:, :, trainInd); trainY = ys(trainInd); %Training set

    %% 0.2 Set hyperparameters
    n1 = 30; k1 = 5; %Num & width of filters at layer 1
    n2 = 30; k2 = 5; %Num & width of filters at layer 2
    n_len1 = n_len - k1 + 1; n_len2 = n_len1 - k2 + 1; widW = n2 * n_len2;
    
    %Weight initialization
    sig1 = sqrt(2/(1*d*n_len)); sig2 = sqrt(2/(n1*n_len1)); sig3 = sqrt(2/widW);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(K, widW)*sig3;
    
    %Momentum vector
    v = cell(3,1); v{1} = zeros(d, k1, n1); v{2} = zeros(n1, k2, n2); v{3} = zeros(K, widW); 
    
    %% 0.5 Balanced dataset
    tabT = tabulate(trainY);
    
    %Option 1: Normalize
%
    tabTpy(:,1) = 1./(tabT(:,2)*0.001); %Norm value %tabVpy(:,1) = 1./(tabV(:,2)*1);
    tabTpy(:,2) = tabT(:,3)./100; tabTpy(:,3) = tabT(:,2); %Ratios & Freq %tabVpy(:,2) = tabV(:,3)./100; %tabVpy(:,3) = tabV(:,2);
    choice = 1; py = tabTpy(:, choice); %pyV = tabVpy(:, choice);
    %Choice: '1' normalize w.r.t occurence, '2' ratios
%}
    %Option 2: Random pick in the size of smallest class samples
%{    
    py = ones(K,1); pick = min(tabT(:,2)); 
    GDparams.n_batch = 3; GDparams.batches = K*pick/GDparams.n_batch;
    set = zeros(2, K); s = 1; e = 0;
    for i=1:K
        e = e + tabT(i,2);
        set(:, i) = [s; e]; s = e+1;
    end
%}    
    %Unbalanced dataset
    GDparams.n_batch = 19; GDparams.batches = trainN/GDparams.n_batch;
    
    GDparams.eta = 0.001; GDparams.rho = 0.9; 
    GDparams.n_epochs = ceil(30000/GDparams.batches);
    %divisors(19798): 2, 19, 38, 521, 1042, 9899, 19798; divisors(18*59): 2, 3, 6, 9, 18, 59, 118, 1062   
    %% 0.3 Construct the convolution matrices
%
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
    
    %% Training
    disp('***Training begins!***'); 
    sprintf('Number of epochs: %d\n Batch size: %d, Batches: %d, updates: %d',GDparams.n_epochs, GDparams.n_batch, GDparams.batches, GDparams.n_batch*GDparams.batches)
    
    %Initialization
    J_train = zeros(GDparams.n_epochs, 1); J_val = zeros(GDparams.n_epochs, 1); %Cost
    tA = zeros(GDparams.n_epochs, 1); vA = zeros(GDparams.n_epochs, 1); %Accuracy
    time = 0;  t = 0; %Training time and update count
    
    %Pre-computes
    VX = MakeVecXMatrix(trainX, d, k1);
    
    for e = 1:GDparams.n_epochs %Epochs
        EpochTime = tic; rng(400);
        
        %Random pick and shuffle
%{
        %Balanced dataset Option 02
        index = zeros(K*pick, 1);
        for i=1:K
            c = pick*i; b = c-pick+1;
            index(b:c,1) = randsample(set(1,i):set(2,i),pick)';
        end
        shuffle = index(randperm(length(index)));
        rtrainX = trainX(:,:,shuffle); rtrainY = trainY(shuffle); rVX = VX(:,:,shuffle); 
%}
        %Unbalanced dataset or Option 01
        shuffle = randperm(trainN);
        rtrainX = trainX(:,:,shuffle); rtrainY = trainY(shuffle); rVX = VX(:,:,shuffle);
        
        %Batchwise parameter updation
        ord = randperm(GDparams.batches); %Random shuffle of batches

        for j=1:GDparams.batches 
            %uTime = tic; %Update time
            t = t + 1; % Increment update count
            j_end = ord(j)*GDparams.n_batch;
            j_start = j_end-GDparams.n_batch +1;
            inds = j_start:j_end;
            Xbatch = rtrainX(:,:,inds);
            VXbatch = rVX(:,:,inds); Ybatch = rtrainY(inds);

            %Updates
            %GDparams.eta = cyclic(t, ns, etaMax, etaMin); n(t) = GDparams.eta;
            [ConvNet, ~, v] = MiniBatchGD(Xbatch, VXbatch, Ybatch, ConvNet, GDparams, v, py);
        end
        
        %Evaluate losses
        [J_train(e), YpredT] = ComputeLoss(trainX, trainY, ConvNet, py);
        [J_val(e), YpredV] = ComputeLoss(valX, valY, ConvNet, ones(K,1));
        
        %Accuracy
        tA(e) = 100 * mean(trainY == YpredT);
        vA(e) = 100 * mean(valY == YpredV);
        
        %figure(3); imagesc(confusionmat(valY, YpredV)); title('Confusion Matrix'); 
        
        sprintf('Epoch %d in %d, Total updates %d', e, toc(EpochTime), t)
        time=time+toc(EpochTime);
    end
    sprintf('Training time is %d', time)
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
        xlim([1 e]);  ylim([min(min(tA), min(vA)) max(max(tA), max(max(vA)))]);
        title('Accuracy Plot'); xlabel('Epoch'); ylabel('Accuracy'); grid on;
        legend({'Training','Validation'},'Location','northeast');
        sprintf('Total number of update steps is %2d', t);
%} 
    %figure(3); conf = plotconfusion(valY, YpredV); %save('conf.mat', 'conf');
    %C = confusionmat(valY, YpredV);  T = valY; P = YpredV; save('confusion.mat', 'C', 'T', 'P')
    %C = imagesc(confusionmat(valY, YpredV)); title('Confusion Matrix of validation set'); 
%
    %For test purpose
    test = ["huang"; "maria"; "akeel"; "andreas"; "gonzales"; "woods"];
    testY = [2, 15, 1, 7, 17, 5]'; testX = zeros(d, n_len, length(testY));
    for i = 1:6
        tem = num2str(test(i));
        n_name = length(tem);
        for j = 1: n_name
            ind = char_to_ind(tem(j));
            testX(:, j, i) = bsxfun(@eq, 1:d, ind)'; %one-hot encoding
        end
    end
    [Ypredict, FP, ~] = ForwardPass(testX, ConvNet); 
    A = 100 * mean(testY == Ypredict);
    save('ConvNet.mat', 'ConvNet'); save('Test.mat','testX','testY','Ypredict','FP');
%}    
    sprintf('Total time is %d', toc(TotalTime)); 
end

function [J, Ypred] = ComputeLoss(Xbatch, Ybatch, ConvNet, py)
    %Compute Loss
    %X_batch: n_len*d x n
    [Ypred, FP, ~] = ForwardPass(Xbatch, ConvNet);
    
    Loss = 0; samples = size(Xbatch,3);
    for i=1:samples
        p = FP{3,i};
        Loss = Loss - py(Ybatch(i))*log(p(Ybatch(i)));
    end
    J = Loss ./ samples;
end

function [ConvNetStar, FP, v] = MiniBatchGD(Xbatch, VXbatch, Ybatch, ConvNet, GDparams, v, py)
%Mini batch Gradient Descent Algo
    %Predict
    [~, FP, MF] = ForwardPass(Xbatch, ConvNet);
 
    %Compute gradient
    [gradConvNet] = CompGradients(VXbatch, Ybatch, ConvNet, FP, MF, py);
%
    %Sanity check for gradient
    [gn] = NumericalGradient(Xbatch, Ybatch, ConvNet, 1e-5);
    relerr.W = rerr(gradConvNet.W, gn{3});
    relerr.F{1} = mean(rerr(gradConvNet.F{1}, gn{1})); 
    relerr.F{2} = mean(rerr(gradConvNet.F{2}, gn{2}));
%}    
    %Update the parameters in theta
    v{1} = GDparams.rho*v{1} + GDparams.eta * gradConvNet.F{1};
    v{2} = GDparams.rho*v{2} + GDparams.eta * gradConvNet.F{2};
    v{3} = GDparams.rho*v{3} + GDparams.eta * gradConvNet.W;
    ConvNetStar.F{1} = ConvNet.F{1} - v{1};
    ConvNetStar.F{2} = ConvNet.F{2} - v{2};
    ConvNetStar.W = ConvNet.W - v{3};
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

function [gradConvNet] = CompGradients(VXbatch, Ybatch, ConvNet, FP, MF, py)
    %Compute gradients via back propagation w.r.t  vec(F1), vec(F2), W
    %Xbatch: d x n_len x samples, Ybatch: samples, ConvNet: F1, F2, W
    %FP: 3xsamples of X, MF: 1x2 of F1, F2
    
    [k, w] = size(ConvNet.W); LW = zeros([k, w]);
    [n1, k2, n2] = size(ConvNet.F{2}); LF2 = zeros([n1, k2, n2]);
    [d, k1, n1] = size(ConvNet.F{1}); LF1 = zeros([d, k1, n1]);
    samples = size(Ybatch,1); n_len2 = w./n2; n_len1 = n_len2+k2-1;

    for i = 1:samples
        Yhot = bsxfun(@eq, 1:k, Ybatch(i))';
        %Softmax
        G3 = -(Yhot - FP{3, i});
        %Layer 3
        LW = LW + py(Ybatch(i))*(G3 * FP{2,i}(:)'); %W*X: 18x1 X 1x55 = 18x55
        
        %Layer 2
        G2 = ConvNet.W' * G3; %W*X: 55x18 X 18x1 = 55x1
        G2 = G2 .* (FP{2,i}(:)>0); %ReLU:  55x1 .* 55x1 = 55x1

        VX = MakeVecXMatrix(FP{1,i}, n1, k2); %11x25
        Gp = reshape(G2, [n2, n_len2]); %5x11
        v1 = VX' * Gp'; %11x25' X 5x11' = 25x5
        %MX = MakeMXMatrix(FP{1,i}, n1, k2, n2); %55x125 !
        %v1 = G2' * MX; %isequal(v1(:), v1_check'); !
        LF2 = LF2 + py(Ybatch(i))*reshape(v1, size(LF2));
        
        %Layer 1
        G1 = MF{2}' * G2; %W*X: 75x55 X 55x1 = 75x1
        G1 = G1 .* (FP{1,i}(:)>0); %ReLU:  75x1 .* 75x1 = 75x1
        
        Gp = reshape(G1, [n1, n_len1]); %5x15
        v2 = VXbatch(:,:,i)' * Gp'; %15x140' X 5x15' = 140x5
        %MX{2} = MakeMXMatrix(Xbatch(:,:,i), d, k1, n1);
        %MXpre = VXtoMX(VXbatch(:,:,i), n1);v2 = G1' * MXpre;%isequal(v2c(:), v2'); !
        LF1 = LF1 + py(Ybatch(i))*reshape(v2, size(LF1));
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

%% ==========================================================
function [rerr] = rerr(ga, gn)
    %Compute relative error
    rerr = sum(sum(abs(ga - gn)./max(eps, abs(ga) + abs(gn))))./ numel(ga);
end

function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)
K = size(ConvNet.W, 1);
try_ConvNet = ConvNet;
Gs = cell(length(ConvNet.F)+1, 1);

for l=1:length(ConvNet.F)
    try_convNet.F{l} = ConvNet.F{l};
    
    Gs{l} = zeros(size(ConvNet.F{l}));
    nf = size(ConvNet.F{l},  3);
    
    for i = 1:nf        
        try_ConvNet.F{l} = ConvNet.F{l};
        F_try = squeeze(ConvNet.F{l}(:, :, i));
        G = zeros(numel(F_try), 1);
        
        for j=1:numel(F_try)
            F_try1 = F_try;
            F_try1(j) = F_try(j) - h;
            try_ConvNet.F{l}(:, :, i) = F_try1; 
            
            l1 = ComputeLoss(X_inputs, Ys, try_ConvNet, ones(K,1));
            
            F_try2 = F_try;
            F_try2(j) = F_try(j) + h;            
            
            try_ConvNet.F{l}(:, :, i) = F_try2;
            l2 = ComputeLoss(X_inputs, Ys, try_ConvNet, ones(K,1));            
            
            G(j) = (l2 - l1) / (2*h);
            try_ConvNet.F{l}(:, :, i) = F_try;
        end
        Gs{l}(:, :, i) = reshape(G, size(F_try));
    end
end

%% compute the gradient for the fully connected layer
W_try = ConvNet.W;
G = zeros(numel(W_try), 1);
for j=1:numel(W_try)
    W_try1 = W_try;
    W_try1(j) = W_try(j) - h;
    try_ConvNet.W = W_try1; 
            
    l1 = ComputeLoss(X_inputs, Ys, try_ConvNet, ones(K,1));
            
    W_try2 = W_try;
    W_try2(j) = W_try(j) + h;            
            
    try_ConvNet.W = W_try2;
    l2 = ComputeLoss(X_inputs, Ys, try_ConvNet, ones(K,1));            
            
    G(j) = (l2 - l1) / (2*h);
    try_ConvNet.W = W_try;
end
Gs{end} = reshape(G, size(W_try));
end
function A = ComputeAccuracy(Ybatch, Ypred)
    %Compute the accuracy
    A = mean((valY == YpredV));
end
function MX = VXtoMX(VX, nf)
    MX = zeros(size(VX)*nf); %Not used
    s = 1; e = nf;
    for i=1:size(VX,1)
        MX(s:e, :) = kron(eye(nf), VX(i,:));
        %MX(s:e, :) = MX(s:e, :) + circshift([VF, zeros(nf, (n_len-k)*dd)], [0 shift]) VX(i,:);
        s = s + nf; e = e + nf;
    end
end
function MX = MakeMXMatrix(x_in, d, k, nf)
    %MX: (n_len-k+1)*nf X k*nf*d %Not used
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