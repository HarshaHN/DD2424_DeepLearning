function [P, out] = GenFP(RNN, in)
    at = RNN.W*in.ht + RNN.U(:, in.x) + RNN.b; %mx1: mxm X mx1 + mxK X Kx1
    out.ht = tanh(at); %mx1 
    ot = RNN.V*out.ht + RNN.c; %Kx1: Kxm X mx1
    P = softmax(ot); %Kx1
    [~, out.x] = min(P);
end