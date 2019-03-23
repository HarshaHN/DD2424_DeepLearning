%addpath ./Datasets/cifar-10-batches-mat/;
A = load('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
%A.data and A.labels;
I = reshape(A.data', 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
montage(I(:, :, :, 1:500), 'Size', [5,5]);
P = EvaluateClassifier(trainX(:, 1:100), W, b);