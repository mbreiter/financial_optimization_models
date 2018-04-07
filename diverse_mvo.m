function x_optimal = diverse_mvo(mu, Q, targetRet, card)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
  n = size(Q,1);
  rho = corrcov(Q);

  clear model;
  clear params;

  model.Q = sparse(zeros(power(n,2) + n));
  model.obj = -1 * cat(1, rho(:), zeros(n,1));

  size_constraint = cat(2, zeros(1, power(n,2)), ones(1,n));

  % Setting up the one-rep-per-asset constraint
  A = ones(1,n);
  N = n;
  Ar = repmat(A, 1, N);
  Ac = mat2cell(Ar, size(A,1), repmat(size(A,2),1,N));

  one_rep_constraint = cat(2, blkdiag(Ac{:}), zeros(n));

  % Setting up the reps-must-be-in-portfolio constraint
  A = -1 * ones(n,1);
  N = n; 
  Ar = repmat(A, 1, N);
  Ac = mat2cell(Ar, size(A,1), repmat(size(A,2),1,N));

  reps_in_portfolio_constraint = cat(2, eye(power(n,2)), blkdiag(Ac{:}));

  model.A = sparse(cat(1, size_constraint, one_rep_constraint, reps_in_portfolio_constraint));
  model.sense = cat(1, repmat('=', n+1, 1), repmat('<', power(n,2), 1));
  model.rhs = cat(1, card, ones(n,1), zeros(power(n,2), 1));
  model.vtype = repmat('B', power(n,2) + n, 1);
  
  params.outputflag = 1;

  result = gurobi(model, params);

  result_x = result.x;
  assets_in_portfolio = result_x(power(n,2) + 1, power(n,2) + n);

  % Creating the returns and covariances of the assets that will be in the portfolio
  portfolio_Q = [];
  portfolio_mu = [];

  for i = 1:n
    temp_values = [];
    for j = 1:n
      if assets_in_portfolio(j) == 1
        temp_values = cat(2, temp_values, Q(i,j));
      end
    end
    if assets_in_portfolio(i) == 1
      portfolio_mu = cat(1, portfolio_mu, mu(i));
      portfolio_Q = cat(1, portfolio_Q, temp_values);
    end
  end
  
  % Get the optimal portfolio from the MVO function
  x_optimal = MVO(portfolio_mu, portfolio_Q, targetRet);

end

