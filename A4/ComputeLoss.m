function [J, Ypred] = ComputeLoss(RNN, X, Y, h0)

    %h0 = zeros();
    [P, ~, Ypred] = ForwardPass(RNN, X, h0);
    
    Loss = 0; seq = length(X);
    for i = 1:seq
        pt = P(:,i); yt = Y(i);
        Loss = Loss - log(pt(yt));
    end
    J = Loss ./ seq;
end