function [mu, sigma, pi] = initialize_GMM(K, data_dim, data_max, data_min)
% Initializes mu, sigma, pi for a GMM


    % INITIALIZATION
    mu = (rand(K, data_dim) .* (data_max - data_min)) + data_min;
    % Initialize sigmas to identity matrix
    sigma = zeros(K, data_dim, data_dim);
    for i = 1 : K
        sigma(i, :, :) = eye(data_dim, data_dim);
    end
    % Initialize weights uniformly
    pi = (1/K)*ones(K, 1);

end

