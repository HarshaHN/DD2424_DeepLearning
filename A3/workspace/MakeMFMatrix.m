function MF = MakeMFMatrix(F, n_len)
    % MF: (nlen-k+1)*nf X nlen*dd
    [dd, k, nf] = size(F);
    MF = zeros((n_len-k+1)*nf, n_len*dd);
    %VF update
    VF = zeros(nf, dd*k);
    for i = 1:nf
        temp = F(:, :, i);
        VF(i, :) = temp(:)'; %nf X dd*k
    end
    %MF update
    s = 1; e = nf;
    for j = 1:(n_len-k+1)
        shift = (j-1)*dd;
        MF(s:e, :) = MF(s:e, :) + circshift([VF, zeros(nf, (n_len-k)*dd)], [0 shift]);
        s = s + nf; e = e + nf;
    end
end

