function robust_optimal = robust_mvo(mu, Q, lambda, alpha)

    % the number of assets
    N = size(mu,1);
    
    % our covariance matrix with buffer for extra parameter introduced for
    % robust mvo
    Q_r = [Q zeros(N,1); zeros(1,N+1)];
    
    % our standard error term
    Theta = diag(diag(Q)) / N;
    
    % estimate our scaling term from a chi-squared distribution
    epsilon = sqrt(chi2inv(alpha,N));
    
    clear model
    % define our objective functon value and the sense of our optimization
    model.obj = [-mu' epsilon];
    model.Q = sparse(lambda*Q_r);
    model.modelsense = 'min';
    
    % budget constraint
    model.A = sparse([ones(1,N) 0]);
    model.rhs = 1;
    model.sense = '=';

    % handle our quadratic constaint which relates auxillary variable to
    % standard error of returns
    model.quadcon(1).Qc = sparse([Theta zeros(N,1); zeros(1,N) -1]);
    model.quadcon(1).q = zeros(N+1,1);
    model.quadcon(1).rhs = 0;
    model.quadcon(1).sense = '=';
    
    % we permit short-selling on our holding assets
    model.lb = [-100000*ones(N,1); 0];

    result = gurobi(model);
    robust_optimal = result.x(1:N);
end

