function [rerr] = rerr(ga, gn)
%RELERROR Summary of this function goes here
%   Detailed explanation goes here
    rerr = sum(sum(abs(ga - gn)./max(eps, abs(ga) + abs(gn))))./ numel(ga);
end

