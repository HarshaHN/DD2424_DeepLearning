function J = ComputeCost(X, Y, W, b, lambda)
    %COMPUTECOST Summary of this function goes here
    %   Detailed explanation goes here
    % Y: KxN, X: dxN, W: W: Kxd, b: Kx1, lambda
    
    P = EvaluateClassifier(X, W, b); %KxN
    L = -log(Y' * P); %NxN
    totalLoss = trace(L); %sum(diag(L))
    R = sumsqr(W); %sum(sum(W.*W));
    J = (totalLoss)./ size(X, 2) + lambda.*R;

end

