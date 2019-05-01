function [gradRNN] = CompGradients(RNN, X, Y, P, H, A)
    %Compute gradients via back propagation w.r.t  RNN: b, c, U, V, W params
    seq = length(X); m = size(H, 2); K = length(RNN.c);
    LV = zeros(size(RNN.V)); Lc = zeros(size(RNN.c)); Gn = zeros(1,m);
    LW = zeros(size(RNN.W)); LU = zeros(size(RNN.U)); Lb = zeros(size(RNN.b));
    
    for i = seq:1
        Yhot = bsxfun(@eq, 1:K, Y(i))';
        %Softmax
        G = -(Yhot - P(:, i))';
        %Output layer
        LV = LV + (G' * H(:,i+1)'); %83*100: 83x1 X 1x100 = 18x55
        Lc = Lc + G'; % 83x1
        
        %Hidden layer
        G = G * RNN.V + Gn * RNN.W; %1x100: 1x83 X 83x100 + 1x100 X 100x100
        Gn = G * diag(1-power(tanh(A(:,i)),2)); %1x100: 1x100 X 100x100;
        Lb = Lb + Gn'; %100x100
        LW = LW + Gn' * H(:,i)'; %100x100: 100x1 X 1x100
        %Xhot = bsxfun(@eq, 1:K, X(i))'; % Gn' * Xhot'
        LU(:, X(i)) = LU(:, X(i)) + Gn'; %100x83: 1x100' X 83x1'
    end
    gradRNN.b = (1/seq)*Lb; gradRNN.c = (1/seq)*Lc;
    gradRNN.U = (1/seq)*LU; gradRNN.V = (1/seq)*LV; gradRNN.W = (1/seq)*LW; 
end