function [W, b] = InitParam(k, d)
%INITPARAM Summary of this function goes here
    % Initialize the parameters of the model W and b
    W = zeros(k,d); b = zeros(k,1);
    rng(400);
    for i=1:10
        W(i,:) = 0.01.*randn(1,3072);
    end
    b = 0.01.*randn(10,1); %W = 0.01.*randn(10,3072);
    % W: 10x3072, b: 10x1

end

