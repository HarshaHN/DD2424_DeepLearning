function acc = ComputeAccuracy(X, y, W, b)
    %COMPUTEACCURACY Summary of this function goes here
    %   Detailed explanation goes here
    % y: 1xN, X: dxN, W: Kxd, b: Kx1, lambda
    
    P = EvaluateClassifier(X, W, b); %KxN
    [~, argmax] = max(P);
    c = (argmax == y);
    acc = sum(c)/size(c,2);
end
