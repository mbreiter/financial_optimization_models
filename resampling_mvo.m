function [weights_avg] = resampling_mvo(mu, Q, targetRet, T, NoEpisodes)
%	MVO optimization based on resampling techniques. 
%
%	The generated returns used in optimization are generated from a multivariate normal distribution with sample mu and Q. 
%	A total of T returns are generated and the robust mvo optimizer takes in the geometric mean of mu and the sample Q of the generated returns.
%
%	inputs: mu - sample average vector
%			Q - sample covariance matrix
% 			targetRet - target return scalar
% 			T - number of samples to generate from scenarios
%			NoEpisodes - number of episodes the resampling process runs for
%
%	outputs:weights_avg - arithmetic mean of all weights generated from resampling.

for i = 1:NoEpisodes
    ret_gen{i}(:,:) = mvnrnd(mu, Q, T);
    mu_gen{i}(:,:) = geomean(ret_gen{i}+1)-1;
    Q_gen{i}(:,:) = cov(ret_gen{i});
    weights_gen(:,i) = robust_mvo(mu_gen{i}', Q_gen{i}, targetRet); % each col is one sample
end

% Weights averaged using arithmetic mean
weights_avg = mean(weights_gen, 2);

end