function [Wstar, bstar, J] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    %MINIBATCHGD Summary of this function goes here
    %   Detailed explanation goes here
    
    %Compute cost
    P = EvaluateClassifier(X, W, b);
    J = ComputeCost(X, Y, W, b, lambda);
    %Compute gradient
    [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);
    %Update the parameters W, b
    Wstar = W - GDparams.eta * grad_W;
    bstar = b - GDparams.eta * grad_b;
    
end
