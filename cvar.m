function [x_optimal, CVaR] = cvar(mu, Q, prices, beta)
    
    % we need the correlation matrix to simulate the correlated prices of 
    % portfolio
    rho = corrcov(Q);
    
    % we take the cholesky factorization of the correlation matrix
    L = chol(rho, 'lower');
    
    % the number of monte carlo simulations that we will run
    S = 2000;
    
    % the number of assets in our portfolio 
    N = size(prices, 1);
    
    % our simulated asset prices and returns
    sim_price = zeros(N,S);
    sim_returns = zeros(N,S);
        
    % we have weekly estimates for returns and we wish to simulate the
    % price path after six months using a single time-step
    dt = 26;
       
    for i=1:S
        % our random correlated pertubations
        epsilon = L * normrnd(0,1,[N,1]);
        
        % calculate our simulated prices
        sim_price(:,i) = prices .* exp((mu - 0.5 * diag(Q))*dt + sqrt(dt)* diag(Q) .* epsilon);  
        
        % calculate our simulated returns
        sim_returns(:,i) = (sim_price(:,i) - prices) ./ prices;
    end
            
    % we formulate our linear objective function
    % NOTE: ordering of decision variables: gamma, x_1 ... x_N, z_1 ... z_S
    
    % objective function value
    f = [1 zeros(1,N) 1/((1-beta)*S)*ones(1,S)]';
    
    % inequality constraints from our CVaR parameterization
    % -r_s*x - gamma - z_s <= 0
    A = -1*[ones(S,1) sim_returns' eye(S)];
    b = zeros(S,1);
    
    % equality constaints only for budget requirement
    Aeq = [0 ones(1,N) zeros(1,S)];
    beq = 1;
    
    % we prohibit short-selling so all decision-variables are positive
    lb = zeros(1+N+S,1);
    
    % no upper-bound on our decision-variables
    ub = [];
    
    % optimal solution to the linear-program minimizing CVaR
    optimal = linprog(f, A, b, Aeq, beq, lb, ub);
    
    CVaR = optimal(1);
    x_optimal = optimal(2:N+1); 
end

