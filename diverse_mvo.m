function x_optimal = diverse_mvo(mu, Q, targetRet, card)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
n = size(Q,1);
rho = corrcov(Q);

clear model;
model.Q = [];
model.obj = cat(1, rho(:),zeros(n,1));

size_constraint = cat(2, zeros(1, pow(n,2)), ones(1,n));

% Setting up the one-rep-per-asset constraint
A = ones(1,n);
N = n;
Ar = repmat(A, 1, N);
Ac = mat2cell(Ar, size(A,1), repmat(size(A,2),1,N));

one_rep_constraint = blkdiag(Ac{:});

% Setting up the reps-must-be-in-portfolio constraint
A = -1 * ones(n,1);
N = n; 
Ar = repmat(A, 1, N);
Ac = mat2cell(Ar, size(A,1), repmat(size(A,2),1,N));

reps_in_portfolio_constraint = cat(2, eye(pow(n,2)), blkdiag(Ac{:});

model.A = sparse(cat(1, size_constraint, one_rep_constraint, reps_in_portfolio_constraint));
model.sense = 


end

