% DD2424 Deep Learning in Data Science from Prof. Josephine Sullivan
% 03 Assignment dated April 17 2019 
% Author: Harsha HN harshahn@kth.se
% Character-level Convolutional Networks for Text Classification

function three
    close all; clear; clc; tic

    load('DebugInfo.mat');
    load('assignment3_names.mat');

    %%0.0 Data preprocessing
    ExtractNames;
    C = unique(cell2mat(all_names));%List of uniq characters
    d = numel(C); %Num of uniq characters

    k = length(unique(ys)); %Number of unique classes
    N = length(ys);% Num of names in the dataset
    n_len = 0; %Maximum length
    for i = 1: length(names)
        n_len = max(length(all_names{i}), n_len);
    end

    % Character to Index mapping
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    for i = 1: length(C)
        char_to_ind(C(i)) = i;
    end

    % Input (names) encoding into dxn_len
    Y = zeros(d, n_len, N);
    for i = 1: length(names)
        temp = all_names{i};
        n_name = length(temp);
        %Y(:,:,i) = zeros(d,n_len);
        for j = 1: n_name
            ind = char_to_ind(temp(j));
            Y(:, j, i) = bsxfun(@eq, 1:d, ind)'; %one-hot encoding
        end
    end

    %% 0.1 Partition into Train data and Validation data
    data_fname = 'Validation_Inds.txt';
    fid = fopen(data_fname,'r');
    S = fscanf(fid,'%c');
    fclose(fid);
    names = strsplit(S, ' ');

    valN = size(names,2); valInd = zeros(1, valN);
    for i = 1:valN
        valInd(i) = str2double(names{i});
    end
    trainInd = setdiff(1:N,valInd);
    valD = Y(:, :, valInd); trainD = Y(:, :, trainInd);

    %% 0.2 Set hyperparameters
    n1 = 4; %Num of filters at layer 1
    k1 = 5; %width of filter at layer 1
    n2 = 4; %Num of filters at layer 2
    k2 = 5; %width of filter at layer 2
    n_len1 = n_len - k1 + 1; n_len2 = n_len1 - k2 + 1; fsize = n2 * n_len2;
    eta = 0.001; rho = 0.9;
    sig1 = sqrt(2/(1*k1)); sig2 = sqrt(2/(n1*k2)); sig3 = sqrt(2/fsize);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(k, fsize)*sig3;

    %% 0.3 Construct the convolution matrices
 
    %To transfer between the matrix and vector encodings
    %x_input = X_input(:);
    %X_input = reshape(x_input, [d, nlen]);
    x_input = trainD(:, :, 5);
    
    %Convolution into matrix multiplication
    MF = MakeMFMatrix(ConvNet.F{1}, n_len);
    MX = MakeMXMatrix(x_input, d, k1, n1);
        
    s1 = MX * ConvNet.F{1}(:); 
    s2 = MF * x_input(:);    

    toc
end


function MX = MakeMXMatrix(x_in, d, k, nf)
    %MX: (n_len-k+1)*nf X k*nf*d
    [d, n_len] = size(x_in);
    MX = zeros((n_len-k+1)*nf, k*nf*d);
    VF = zeros(n_len-k+1, d*k);
    
    s = 1; e = nf;
    for i=1:(n_len-k+1)
        temp = x_in(:, i:k+i-1);
        VF(i, :) = temp(:)';
        MX(s:e, :) = kron(eye(nf), VF(i, :));
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
