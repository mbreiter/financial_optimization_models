function  x_optimal = mvo(mu, Q, targetRet)

    % Find the total number of assets
    n = size(Q,1);
    
    % f is a vector of zeros, since MVO only has the quadratic terms from Q
    f = zeros(n,1);
    
    % A and b will be used for the constraint for our targetted return. 
    A = -mu';
    b = -targetRet;
    
    % Aeq and beq will house our equality constraint, ensuring that the
    % proprotions in our portifolio sum to 1
    Aeq = ones(1,n);
    beq = 1;  
    
    % We allow shortselling and otherwise have no bounding constraints
    ub = [];
    lb = [];
   
    % We use quadprog to optimize
    x_optimal = quadprog(Q,f,A,b,Aeq,beq,lb,ub);    
end