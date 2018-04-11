function [sim_returns, sim_price] = monte_carlo(mu, Q, prices, S, dt)
    % we need the correlation matrix to simulate the correlated prices of 
    % portfolio
    rho = corrcov(Q);
    
    % we take the cholesky factorization of the correlation matrix
    L = chol(rho, 'lower');
    
    % the number of assets in our portfolio 
    N = size(prices, 1);
    
    % our simulated asset prices and returns
    sim_price = zeros(N,S);
    sim_returns = zeros(N,S);
       
    for i=1:S
        % our random correlated pertubations
        epsilon = L * normrnd(0,1,[N,1]);
        
        % calculate our simulated prices
        sim_price(:,i) = prices .* exp((mu - 0.5 * diag(Q))*dt + sqrt(dt)*sqrt(diag(Q)) .* epsilon);  
        
        % calculate our simulated returns
        sim_returns(:,i) = (sim_price(:,i) - prices) ./ prices;
    end
end

