function P = EvaluateClassifier(X, W, b)

    %SOFTMAX(s) = exp(s)/1T exp(s);
    s = W*X + b; % Kxd*dxN + Kx1 = KxN
    P = softmax(s); % KxN

end