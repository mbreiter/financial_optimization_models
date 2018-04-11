function [x_optimal, VaR, CVaR] = cvar(prices, S, beta, sim_returns, sim_prices)
    
    % the number of assets in our portfolio 
    N = size(prices, 1);
    
      % uncomment if you want to create a mesh plot for your simulated asset prices  
%     X = 1:N;
%     Y = 1:S;
%     mesh(sim_price);
%     title('Simulated Prices of Holding Assets', 'FontSize', 14)
%     ylabel('Asset','interpreter','latex','FontSize',12);
%     xlabel('Scenario','interpreter','latex','FontSize',12);
%     zlabel('Asset Price','interpreter','latex','FontSize',12);
             
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
    lb = [-10e6; zeros(N+S,1)];
    
    % no upper-bound on our decision-variables
    ub = [];
    
    % optimal solution to the linear-program minimizing CVaR
    [optimal,CVaR] = linprog(f, A, b, Aeq, beq, lb, ub);
    
    VaR = optimal(1);
    x_optimal = optimal(2:N+1); 
    
end

