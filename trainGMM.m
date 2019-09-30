function [mu, sigma, pi] = trainGMM(mu, sigma, pi, train_data, K, max_iters, epsilon)
% Trains a GMM modeling data distribution as a sum of K Gaussians 
% Parameters for model i are are mu(i, :) and sigma(i, :, :), 
% Gaussians are weighted by pi(i)
%
% Assumes train_data has shape (n_examples, data_dim)
%
%   [mu, sigma, pi] = trainGMM(train_data, K, max_iters, epsilon)
%       
%       (mu, sigma, pi) parameterize GMM
%       mu has shape (K, data_dim)
%       sigma has shape (K, data_dim, data_dim)
%       pi has shape (K)
    
    train_data = double(train_data);
    
    n_examples = size(train_data, 1);
    data_dim = size(train_data, 2);

    mu_old = mu + 100;
    
    % TRAINING LOOP
    iteration = 0;
  
    while iteration <= max_iters && norm(mu - mu_old) > epsilon
        % EXPECTATION - Compute posterior for each point
        % post has shape (n_examples, K)
        post = computePosterior(mu, sigma, pi, train_data, false);
        
        % MAXIMIZATION - update params
        mu_old = mu;
        % Pre-compute non-normalized priors for use during updates 
        pi = sum(post, 1);
        mu = zeros(K, data_dim);
        for k = 1:K
            sigma(k, :, :) = zeros(data_dim, data_dim);
            for i = 1:n_examples
                % Mean Update
                mu(k, :) = mu(k, :) + (train_data(i,:) .* post(i, k)); 
                % Covariance Update
                centered_data = train_data(i, :) - mu_old(k, :);
                sigma(k, :, :) = squeeze(sigma(k, :, :)) + (centered_data'*centered_data) .* post(i, k);
            end
            mu(k, :) = mu(k, :) / pi(k);
            sigma(k, :, :) = sigma(k, :, :) / pi(k);
        end
        % Normalize pre-computed pi
        pi = (1 / n_examples) * pi;

        iteration = iteration + 1
    end
    
end