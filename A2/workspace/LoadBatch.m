function [X, Y, y] = LoadBatch(filename)

    A = load(filename); 
    X = double(A.data')./255;% dxN 3072x10,000
    Y = bsxfun(@eq, 1:10, A.labels+1)';% KxN 10x10,000
    y = (A.labels + 1)';% 1xN 1x10,000

end