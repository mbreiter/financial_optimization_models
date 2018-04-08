%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Financial Optimization Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the purpose of this program is to test the out-of-sample performance of 
% 5 different portfolio optimization models. We will test the following
% models:
% 1. mvo
% 2. robust mvo
% 3. resampling mvo
% 4. most-diverse mvo
% 5. cvar optimization with monte carlo simulations

% authors: Matthew Reiter, Daniel Kecman and Yiqing Lou
% date: april 11, 2018

clc
clear all
format long

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. read input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the stock weekly prices and factors weekly returns
adjClose = readtable('adj_close.csv', 'ReadRowNames', true);
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));

factorRet = readtable('ff_factors.csv', 'ReadRowNames', true);
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));

riskFree = factorRet(:,4);
factorRet = factorRet(:,1:3);

% identify the tickers and the dates 
tickers = adjClose.Properties.VariableNames';
dates   = datetime(factorRet.Properties.RowNames);

% calculate the stocks' weekly EXCESS returns
prices  = table2array(adjClose);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);
returns = returns - ( diag( table2array(riskFree) ) * ones( size(returns) ) );
returns = array2table(returns);
returns.Properties.VariableNames = tickers;
returns.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. define the initial parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of shares outstanding per company
mktNoShares = [36.40 9.86 15.05 10.91 45.44 15.43 43.45 33.26 51.57 45.25 ...
               69.38 8.58 64.56 30.80 36.92 7.70 3.32 54.68 38.99 5.78]' * 1e8;

% initial budget to invest
initialVal = 100;

% robust mvo parameters 
lambda = 50;
alpha = 0.9;
           
% parameters for resampling mvo
T = 100;
NoEpisodes = 100;

% parameters for most-diverse mvo
card = 12;
    
% start of in-sample calibration period 
calStart = datetime('2012-01-01');
calEnd   = calStart + calmonths(12) - days(1);

% start of out-of-sample test period 
testStart = datetime('2013-01-01');
testEnd   = testStart + calmonths(6) - days(1);

% number of investment periods (each investment period is 6 months long)
NoPeriods = 6;

% investment strategies
funNames  = {'mvo' 'robust mvo' 'resampling mvo' 'most-diverse mvo' 'cvar'};
NoMethods = length(funNames);

funList = {'mvo' 'robust_mvo' 'resampling_mvo' 'diverse_mvo' 'cvar'};
funList = cellfun(@str2func, funList, 'UniformOutput', false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. construct and rebalance portfolios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initiate counter for the number of observations per investment period
toDay = 0;

% currentValue keeps track of the value of the portfolio at the beginning
% of each investing period. Value of the portfolio is also tracked daily in
% a variable called portfValue declared later in this section.
currentVal = zeros(NoPeriods, NoMethods);
periodEndVal = zeros(NoPeriods, NoMethods);

% tCost tracks the transcaction costs incurred from rebalancing
tCost = zeros(NoPeriods-1, NoMethods);

% define the following to keep track of what we expect our portfolio to return 
% return per period
expectedReturns = zeros(NoPeriods, NoMethods);
expectedVar = zeros(NoPeriods, NoMethods);

% define the following to keep track of our realized long term rate of
% return. That is, what would be the daily return of the portfolio, benchmarked
% against the first day of the simulation. 
avgReturnsInitial = zeros(NoPeriods, NoMethods);

% define the following to keep track of our per daily rate of returns period 
% benchmarked per period
avgReturnsPeriod = zeros(NoPeriods, NoMethods);

% track the variance of each of our portfolio's value per period
portVarPeriod = zeros(NoPeriods,NoMethods);

for t = 1:NoPeriods    
    % subset the returns and factor returns corresponding to the current calibration period.
    periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
    periodFactRet = table2array( factorRet( calStart <= dates & dates <= calEnd, :) );
    currentPrices = table2array( adjClose( ( calEnd - days(7) ) <= dates ... 
                                                    & dates <= calEnd, :) )';
                                                    
    % subset the prices corresponding to the current out-of-sample test period.
    periodPrices = table2array( adjClose( testStart <= dates & dates <= testEnd,:) );
    
    % update counter for the number of observations per investment period
    fromDay = toDay + 1;
    toDay   = toDay + size(periodPrices,1);
    
    % set the initial value of the portfolio or update the portfolio value
    if t == 1
        currentVal(t,:) = initialVal*ones(1,NoMethods);
    else
        for i = 1 : NoMethods   
            currentVal(t,i) = currentPrices' * NoShares{i};      
        end
    end
     
    % our Fama-French factor model is: R(:,i) = X*beta(:,i) + epsilon
    %   -  X is a size(periodFactRet,1) by 4 matrix,
    %   -  R is equal to periodReturns, 
    %   -  beta is a 4 by 20 vector for our regression coefficients
    %   -  epsilon is a size(periodReturns,1) by size(periodReturns,2)
    %      vector of idiosynratic noise
    
    X = [ones(size(periodFactRet,1), 1) periodFactRet];    
    beta = inv(X'*X)*X'*periodReturns;
        
    % we calculuate our regression residuals and covariances
    epsilon = periodReturns-X*beta;
    D = cov(epsilon);

    % calculating the geometric mean of our factor returns 
    f_bar = [1 (geomean(X(:,2:end)+1) - 1)]';
    F = cov(periodFactRet);
    
    % our expected asset returns
    mu = beta'*f_bar;
    
    % our covariance matrix. Note that here, we do not assume ideal
    % conditions.
    Q = beta(2:end,:)'*F*beta(2:end,:) + D;
        
    % Define the target return for the 2 MVO portfolios
    targetRet = mean(mu);
    
    % optimize each portfolios to get the weights 'x'
    T = 100;
    NoEpisodes = 100;
    
    % optimize each portfolios to get the weights 'x'
    x{1}(:,t) = funList{1}(mu, Q, targetRet);
    x{2}(:,t) = funList{2}(mu, Q, lambda, alpha);
    x{3}(:,t) = funList{3}(mu, Q, targetRet, T, NoEpisodes);
    x{4}(:,t) = funList{4}(mu, Q, targetRet, card);
    [x{5}(:,t), VaR(t)] = funList{5}(mu, Q, currentPrices, 0.95);
    
    for i=1:NoMethods
        % number of shares your portfolio holds per stock
        NoShares{i} = x{i}(:,t) .* currentVal(t,i) ./ currentPrices;
        
        % weekly portfolio value during the out-of-sample window
        portfValue(fromDay:toDay,i) = periodPrices * NoShares{i};
        
        % Calculate your transaction costs for the current rebalance period
        if t ~= 1
            tCost(t-1, i) = 0.005*currentPrices'*abs(NoSharesOld{i} - NoShares{i});        
        end
        
        NoSharesOld{i} = NoShares{i};


        %--------------------- Performance Metrics ----------------------------
        % Ex Ante Sharpe Ratio
        % Expected return of the portfolio based on current weights
        mu_portfolio{i}(t,:) = mu'*x{i}(:,t);
        % Variance of the portfolio based on current weights
        portfolio_var{i}(t,:) = x{i}(:,t)'*Q*x{i}(:,t);
        sharpe_ratio_ante{i}(t,:) = (mu_portfolio{i}(t,:))/sqrt(portfolio_var{i}(t,:));
         
        % Coefficient of Variance
        coefficientOfVariance{i}(t,:) = portfolio_var{i}(t,:)/mu_portfolio{i}(t,:);
        
        % Ex Post Sharpe Ratio
        if t ~= 1
            sharpe_ratio_post{i}(t-1,:) = mu_portfolio_post*x{i}(:,t-1)/sqrt(portfolio_var{i}(t-1,:));%(avgReturnsPeriod(t-1,i))/sqrt(portfolio_var{i}(t-1,:));
        end
        mu_portfolio_post = geomean(periodReturns+1)-1       
    end

    
    % Complete our per period analysis calculations
    periodEndVal(t,:) = portfValue(toDay,:);
    initial_returns = (portfValue(fromDay:toDay,:) - initialVal) / initialVal;
    period_returns = ( portfValue(fromDay:toDay,:) - repmat(currentVal(t,:), size(portfValue(fromDay:toDay,:),1), 1) ) ./ repmat(currentVal(t,:), size(portfValue(fromDay:toDay,:),1), 1);
    
    avgReturnsInitial(t,:) = geomean(initial_returns + 1) - 1;
    avgReturnsPeriod(t,:) = geomean(period_returns + 1) - 1;
    portRiskPeriod(t,:) = std(period_returns);
    
    % update calibration and out-of-sample test periods
    calStart = calStart + calmonths(6);
    calEnd = calStart + calmonths(12) - days(1);
    
    testStart = testStart + calmonths(6);
    testEnd = testStart + calmonths(6) - days(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% my avgReturnsInitial metric provides an average return of the value of the 
% portfolio at the end of each rebalancing period based on the initial value 
% as a benchmark. Used to track what kind of return an investor who invested 
% at the beginning would realize at the end. Note that I calculate this above
avgReturnsInitial;

% my avgReturnsPeriod metric is similar to avgReturnsInitial, but tracks 
% performance based on the value of the portfolio at the beginning of each
% investing period. Tracks the return an investor can expect by investing
% at each period. 
avgReturnsPeriod;

% maximum value of each portfolio and the realization date
[maxVal, when_max] = max(portfValue);
clear max

% minimum value of each portfolio and the realization date
[minVal, when_min] = min(portfValue);
clear min

% calculate weekly differences and find the largest week to week gain, and
% the smallest week to week gain.
difference = portfValue(2:end,:) - portfValue(1:end-1,:);

lwg = zeros(2, NoMethods);
lwl = zeros(2, NoMethods);

for i=1:NoMethods
    [lwg(1,i),lwg(2,i)] = max(difference(:,i));
    [lwl(1,i),lwl(2,i)] = min(difference(:,i));
    
    clear max min
end

% calculate the longest winning and losing streak
streak = zeros(4, NoMethods);
counter = zeros(2,1);

for i=1:NoMethods
    for j=1:size(difference,1)
        if difference(j,i) < 0
            % gain streak is over, so update our gain streak
            if streak(1,i) < counter(1)
                streak(1,i) = counter(1);
                streak(2,i) = j;
            end
            
            counter(1) = 0;
            
            % increment up our loss streak
            counter(2) = counter(2) + 1;
        elseif difference(j,i) > 0
            % loss streak is over, so update our loss streak
            if streak(3,i) < counter(2)
                streak(3,i) = counter(2);
                streak(4,i) = j;

            end
            counter(2) = 0;
            
            % increment up our gain streak
            counter(1) = counter(1) + 1;
        end
    end
    
    counter = zeros(2,1);
end

% %--------------------------------------------------------------------------
% % 4.1 Plot the portfolio values 
% %--------------------------------------------------------------------------

testStart = datetime('2013-01-01');
testEnd = testStart + 6*calmonths(6) - days(1);

plotDates = dates(testStart <= dates);

fig1 = figure(1);

for i=1:NoMethods
    plot(plotDates, portfValue(:,i))
    hold on
end

legend(funNames, 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
title('Portfolio Value', 'FontSize', 14)
ylabel('Value','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

print(fig1,'portfolio-values','-dpng','-r0');

%--------------------------------------------------------------------------
% 4.2 Plot the portfolio weights 
%--------------------------------------------------------------------------

for i=1:NoMethods
    fig = figure(i+1);
    area(x{i}')
    
    if i==1
        name = 'MVO Portfolio Weights';
        file = 'mvo-plot';
    elseif i==2
        name = 'Robust MVO Portfolio Weights';
        file = 'robust-mvo-plot';
    elseif i==3
        name = 'Resampling MVO Portfolio Weights';
        file = 'resampling-mvo-plot';
    elseif i==4
        name = 'Most-Diverse MVO Portfolio Weights';
        file = 'diverse-mvo-plot';
    else
        name = 'CVaR Optimization Portfolio Weights';
        file = 'cvar-plot';
    end
    
    title(name, 'FontSize', 14)
    
    legend(tickers, 'Location', 'eastoutside','FontSize',12);
    ylabel('Weights','interpreter','latex','FontSize',12);
    xlabel('Rebalance Period','interpreter','latex','FontSize',12);

    % Define the plot size in inches
    set(fig,'Units','Inches', 'Position', [0 0 8, 5]);
    pos = get(fig,'Position');
    set(fig,'PaperPositionMode','Auto','PaperUnits','Inches',...
        'PaperSize',[pos(3), pos(4)]);

    % saving the figure as a png
    print(fig,file,'-dpng','-r0');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%