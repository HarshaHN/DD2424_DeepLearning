function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)

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
            
            l1 = Compute_loss(X_inputs, Ys, try_ConvNet);
            
            F_try2 = F_try;
            F_try2(j) = F_try(j) + h;            
            
            try_ConvNet.F{l}(:, :, i) = F_try2;
            l2 = Compute_loss(X_inputs, Ys, try_ConvNet);            
            
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
            
    l1 = Compute_loss(X_inputs, Ys, try_ConvNet);
            
    W_try2 = W_try;
    W_try2(j) = W_try(j) + h;            
            
    try_ConvNet.W = W_try2;
    l2 = Compute_loss(X_inputs, Ys, try_ConvNet);            
            
    G(j) = (l2 - l1) / (2*h);
    try_ConvNet.W = W_try;
end
Gs{end} = reshape(G, size(W_try));
end

function [J, Ypred] = Compute_loss(Xbatch, Ybatch, ConvNet)
    %Compute Loss
    %X_batch: n_len*d x n
    [Ypred, FP, ~] = ForwardPass(Xbatch, ConvNet);
    
    Loss = 0; samples = size(Xbatch,3);
    for i=1:samples
        p = FP{3,i};
        Loss = Loss - log(p(Ybatch(i)));
    end
    J = Loss ./ samples;
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
