% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 04 Assignment dated May 2 2019 
% RNN to synthesize english text character-by-character
%-------------------------------------------------------
%Code Author: Harsha HN
%-------------------------------------------------------

function four
    close all; clear; clc; 
    %TotalTime = tic;
    
    %% 0.1 Read in the data
    bookFname = 'data/Goblet.txt';
    fid = fopen(bookFname, 'r');
    bookData = fscanf(fid, '%c'); fclose(fid);
    bookChars = unique(bookData); K = length(bookChars); bookN = length(bookData);
    
    % Character - index mapping
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');
    for i = 1:K
        char_to_ind(bookChars(i)) = i;
        ind_to_char(i) = bookChars(i);
    end
    
    %% 0.2 Data to vectors
    disp('***Data pre-processing begins***')
    %Character to index
    bookInd = zeros(bookN,1);
    for i = 1:bookN
        bookInd(i, 1) = char_to_ind(bookData(i));
    end
    
    % Input and label
    %trainX = bookInd(1:bookN-1);
    %trainY = bookInd(2:bookN);
%{
    %Input (character) encoding into KxbookN
    trainX_hot = zeros(K, bookN); %h0 = zeros(m, 1);   
    for i = 1:bookN-1
        trainX_hot(:, i) = bsxfun(@eq, 1:K, trainX(i))';
    end
    trainY_hot = [X_hot(:,2:end), bsxfun(@eq, 1:K, trainX(i+1))'];
%}    
    disp('***Data pre-processing completed***')
    
    %% 0.3 Set hyper-parameters & initialize the RNN's parameters
    m = 100; %Dim of hidden state
    GDparams.eta = 0.1; GDparams.n_epochs = 1;
    GDparams.seqlen = 25; GDparams.batches = floor((bookN-1)/GDparams.seqlen);
    GDparams.n_epochs = ceil(300000/GDparams.batches);
    
    %Weight initialization
    sig = .01; RNN.b = zeros(m,1); RNN.c = zeros(K,1);
    RNN.U = randn(m, K)*sig; RNN.W = randn(m, m)*sig; RNN.V = randn(K, m)*sig;
    for f1 = fieldnames(RNN)'
        GDparams.mt.(f1{:}) = 0;
    end
    % 0.4 Synthesize text from your randomly initialized RNN
%{
    h0 = zeros(m, 1); x0 = 'h'; seqlen = 25;
    
    in.ht = h0; in.x = char_to_ind(x0); out = char; out(1) = x0;
    for j = 1:seqlen
        [~, in] = GenFP(RNN, in); out(j+1) = ind_to_char(in.x);
        %ixs = find(cumsum(P) - rand >0); in.x = ixs(1);
    end
    disp(out)
%}    
    
    % 0.5 Implement the forward & backward pass of back-prop
    disp('***Training begins!***'); t = 0; loss_count = 0; %time = tic; 
    smooth_loss = zeros(floor(GDparams.batches/100),1);
    sprintf('Number of epochs: %d\n Seq len: %d\n Batches: %d\n updates: %d',GDparams.n_epochs, GDparams.seqlen, GDparams.batches, GDparams.n_epochs*GDparams.batches)
    %figure(1); title('Loss Plot'); xlabel('x*100 updates'); ylabel('Smooth Loss'); grid on;
    
    for e = 1:GDparams.n_epochs %Epochs
        EpochTime = tic; h = zeros(m, 1);
        
        %Batchwise parameter updation
        for j = 1:GDparams.batches
            t = t + 1; % Increment update count
            j_end = j*GDparams.seqlen;
            j_start = j_end-GDparams.seqlen +1;
            Xbatch = bookInd(j_start:j_end);
            Ybatch = bookInd(j_start+1:j_end+1); h0 = h;
            
            [RNN, GDparams, P, h] = MiniBatchGD(Xbatch, Ybatch, RNN, GDparams, h);
            
            if mod(j,1000) == 0
               loss_count = loss_count + 1;
               [loss] = CompLoss(Ybatch, P); %[loss, ~] = ComputeLossXXX(RNN, Xbatch, Ybatch, P, h0);
               if t == 1000
                   smooth_loss(1) = loss;
               else
                   smooth_loss(loss_count) = .99* smooth_loss(loss_count-1) + .01 * loss;
               end
               %plot(smooth_loss(1:j/100)); axis([1 j/100+1 min(smooth_loss) max(smooth_loss)]); drawnow
               
               if mod(j,1000) == 0 %Synthesis 200 characters
                    seqlen = 25; x = Xbatch(1); seq = char; %GenTime = tic;
                    [out] = GenSeq(RNN, x, h0, seqlen);
                    for i = 1:length(out)
                        seq(i) = ind_to_char(out(i));
                    end
                    disp(seq)
                    sprintf('Smooth loss: %f\n with updates: %d ',smooth_loss(loss_count), t)
                    %sprintf('Gen in %d,\n Num of updates %d time is %d', toc(GenTime), t, time);
               end
            end
        end
        sprintf('Epoch %d in %f,\n Total updates %d', e, toc(EpochTime), t)
        %time=time+toc(EpochTime);
    end
    %% Evaluation
    
    %sprintf('Training time is %d', toc(time))
%{      
    %Plot of Loss
    figure(1); plot(smooth_loss);
    xlim([1 length(smooth_loss)]); ylim([min(smooth_loss) max(smooth_loss)]);
    title('Loss Plot'); xlabel('x*100 updates'); ylabel('Smooth Loss'); grid on;
    sprintf('Total number of update steps is %2d', t);
%}
    sprintf('Total time is %d', toc(TotalTime));     
end

function [RNNstar, GDparams, P, h] = MiniBatchGD(X, Y, RNN, GDparams, h)
    %Mini batch Gradient Descent Algo
    %Predict
    [P, H, A, ~] = ForwardPass(RNN, X, h); h = H(:, end);
 
    %Compute gradient
    [gradRNN] = CompGradients(RNN, X, Y, P, H, A);
%{
    %Sanity check for gradient
    [gn] = ComputeGradsNum(X, Y, RNN, 1e-4);
    for f1 = fieldnames(gradRNN)'
        relerr.(f1{:}) = rerr(gradRNN.(f1{:}), gn.(f1{:}));
    end
%}  
    
    for f1 = fieldnames(gradRNN)'
        %Gradient clipping
        gradRNN.(f1{:}) = max(min(gradRNN.(f1{:}), 5), -5);
        
        %Update the parameters in RNN
        GDparams.mt.(f1{:}) = GDparams.mt.(f1{:}) + power(gradRNN.(f1{:}),2); 
        f = GDparams.eta ./sqrt(GDparams.mt.(f1{:}) + eps); delta = f .* gradRNN.(f1{:});
        %sprintf('%c: %f',f1{:}, f)
        RNNstar.(f1{:}) = RNN.(f1{:}) - delta*10;
    end
end

function [gradRNN] = CompGradients(RNN, X, Y, P, H, A)
    %Compute gradients via back propagation w.r.t  RNN: b, c, U, V, W params
    seq = length(X); m = size(H,1); K = length(RNN.c);
    LV = zeros(size(RNN.V)); Lc = zeros(size(RNN.c)); Gn = zeros(1,m);
    LW = zeros(size(RNN.W)); LU = zeros(size(RNN.U)); Lb = zeros(size(RNN.b));
    
    for i = seq:-1:1
        Yhot = bsxfun(@eq, 1:K, Y(i))';
        %Softmax
        G = -(Yhot - P(:, i))';
        %Output layer
        LV = LV + (G' * H(:,i+1)'); %83*100: 83x1 X 1x100 = 83x100 +1
        Lc = Lc + G'; % 83x1
        
        %Hidden layer
        G = G * RNN.V + Gn * RNN.W; %1x100: 1x83 X 83x100 + 1x100 X 100x100
        Gn = G * diag(1-power(tanh(A(:,i)),2)); %1x100: 1x100 X 100x100;
        Lb = Lb + Gn'; %100x100
        LW = LW + Gn' * H(:,i)'; %100x100: 100x1 X 1x100
        %Xhot = bsxfun(@eq, 1:K, X(i))'; % Gn' * Xhot'
        LU(:, X(i)) = LU(:, X(i)) + Gn'; %100x83: 1x100' X 83x1'
    end
    gradRNN.b = (1/seq)*Lb; gradRNN.c = (1/seq)*Lc;
    gradRNN.U = (1/seq)*LU; gradRNN.V = (1/seq)*LV; gradRNN.W = (1/seq)*LW; 
end

function [P, H, A, O] = ForwardPass(RNN, X, h0)
    seq = length(X); m = length(h0); K = length(RNN.c);
    H = zeros(m, seq+1); P = zeros(K, seq); A = zeros(m, seq); O = zeros(seq, 1);
    
    H(:, 1) = h0;
    for i = 1:seq
        A(:, i) = RNN.W*H(:, i) + RNN.U(:, X(i)) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
        H(:, i+1) = tanh(A(:, i)); %mx1 
        ot = RNN.V*H(:, i+1) + RNN.c; %Kx1: Kxm X mx1
        P(:, i) = softmax(ot); %Kx1
        [~, O(i)] = max(P(:, i));
    end
end

function [J] = ComputeLoss(RNN, X, Y, h0)

    [P, ~, ~, ~] = ForwardPass(RNN, X, h0);
    
    Loss = 0; seq = length(X);
    for i = 1:seq
        pt = P(:,i); yt = Y(i);
        Loss = Loss - log(pt(yt));
    end
    J = Loss ./ seq;
end

function [out] = GenSeq(RNN, x, h, seqlen)
    out=zeros(seqlen,1); out(1) = x;
    for i = 1:seqlen-1
        at = RNN.W*h + RNN.U(:, x) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
        h = tanh(at); %mx1 
        ot = RNN.V*h + RNN.c; %Kx1: Kxm X mx1
        P = softmax(ot); %Kx1
        [~, x] = max(P); out(i+1) = x;
    end
end
                    
function [P, out] = GenFP(RNN, in)
    at = RNN.W*in.ht + RNN.U(:, in.x) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
    out.ht = tanh(at); %mx1 
    ot = RNN.V*out.ht + RNN.c; %Kx1: Kxm X mx1
    P = softmax(ot); %Kx1
    [~, out.x] = max(P);
end

function [J] = CompLoss(Y, P)
    Loss = 0; seq = length(Y);
    for i = 1:seq
        pt = P(:,i); yt = Y(i);
        Loss = Loss - log(pt(yt));
    end
    J = Loss ./ seq;
end

function [rerr] = rerr(ga, gn)
    %Compute relative error
    rerr = sum(sum(abs(ga - gn)./max(eps, abs(ga) + abs(gn))))./ numel(ga);
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);%Slow
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(RNN_try, X, Y, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(RNN_try, X, Y, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end

