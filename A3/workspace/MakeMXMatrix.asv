function MX = MakeMXMatrix(x_in, d, k, nf)
    %MX: (n_len-k+1)*nf X k*nf*d
    [d, n_len] = size(x_in);
    MX = zeros((n_len-k+1)*nf, k*nf*d);
    VF = zeros(n_len-k+1, d*k);
    
    s = 1; e = nf;
    for i=1:(n_len-k+1)
        temp = x_in(:, i:k+i-1);
        VF(i, :) = temp(:)';
        MX(s:e, :) = kron(eye(nf), VF(i, :));
        s = s + nf; e = e + nf;
    end
end
