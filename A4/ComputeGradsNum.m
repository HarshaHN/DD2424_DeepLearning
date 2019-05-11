function num_grads = ComputeGradsNum(X, Y, RNN, hprev, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, hprev, h);%Slow
    end
end
function grad = ComputeGradNum(X, Y, f, RNN, hprev, h)

    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    %hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(RNN_try, X, Y, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(RNN_try, X, Y, hprev);
        grad(i) = (l2-l1)/(2*h);

    end
end

function [J] = ComputeLoss(RNN, X, Y, h0)

    [P, ~, ~, ~] = ForwardPass(RNN, X, h0);
    
    Loss = 0; seq = length(X);
    for i = 1:seq
        pt = P(:,i); yt = Y(i);
        Loss = Loss - log(pt(yt));
    end
    J = Loss ./ seq;
end    

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