function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    %COMPUTEGRADIENTS Summary of this function goes here
    %   Detailed explanation goes here
    % Y or P: KxN, X: dxN, W: Kxd, b: Kx1
    
    %Initialize
    LossW = 0; Lossb = 0;
    
    %Update loop
    N = size(X, 2);
    for i = 1:N % ith image
        g = -(Y(:,i)-P(:,i))'; %NxK
        LossW = LossW + g' * X(:,i)';
        Lossb = Lossb + g';
    end
    
    R = 2*W;
    Jw = (1./size(X, 2)) * (LossW) + lambda.*R;
    Jb = (1./size(X, 2)) * (Lossb);
    
    grad_W = Jw; %Kxd.
    grad_b = Jb; %Kx1
    
end