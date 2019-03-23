function A = ComputeAccuracy(X, y, W, b)
    % y: 1xN, X: dxN, W: Kxd, b: Kx1, lambda    
    P = EvaluateClassifier(X, W, b); %KxN
    [~, argmax] = max(P);
    c = (argmax == y);
    A = sum(c)/size(c,2);
end
