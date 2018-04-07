function x_optimal = diverse_mvo(mu, Q, targetRet, card)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
n = size(Q,1);
rho = corrcov(Q);

clear model;
model.Q = [];
model.obj = cat(1, rho(:),zeros(n,1));
size_constraint = cat(2, zeros(1, pow(n,2)), ones(1,n));
one_rep_per_asset = blkdiag(

end

