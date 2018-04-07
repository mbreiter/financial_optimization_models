function [weights_avg] = resampling_mvo(mu, Q, targetRet, T, NoEpisodes)

for i = 1:NoEpisodes
    ret_gen{i}(:,:) = mvnrnd(mu, Q, T);
    mu_gen{i}(:,:) = geomean(ret_gen{i}+1)-1;
    Q_gen{i}(:,:) = cov(ret_gen{i});
    weights_gen(:,i) = robust_mvo(mu_gen{i}', Q_gen{i}, targetRet);% each col is one sample
end

% Weights averaged using arithmetic mean
weights_avg = mean(weights_gen, 2);

end

