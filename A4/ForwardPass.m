function [P, H, A, O] = ForwardPass(RNN, X, h0)
    seq = length(X); m = length(h0); K = length(RNN.c);
    H = zeros(m, seq+1); P = zeros(K, seq); A = zeros(m, seq); O = zeros(seq, 1);
    
    H(:, 1) = h0;
    for i = 1:seq
        A(:, i) = RNN.W*H(:, i) + RNN.U(:, X(i)) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
        H(:, i+1) = tanh(A(:, i)); %mx1 
        ot = RNN.V*H(:, i+1) + RNN.c; %Kx1: Kxm X mx1
        P(:, i) = softmax(ot); %Kx1
        [~, O(i)] = max(P(:, i));
    end
end

% function [P, out] = GenFP(RNN, in)
%     at = RNN.W*in.ht + RNN.U(:, in.x) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
%     out.ht = tanh(at); %mx1 
%     ot = RNN.V*out.ht + RNN.c; %Kx1: Kxm X mx1
%     P = softmax(ot); %Kx1
%     [~, out.x] = max(P);
% end

