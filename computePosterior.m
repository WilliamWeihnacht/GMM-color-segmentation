function post = computePosterior(mu, sigma, pi, points, thresholding)
% Compute posterior probabilities P(C_i | point) for each of
% the K Gaussians paramaterized by (mu, sigma)
%
%   post = computePosterior(mu, sigma, pi, points)
%       mu = (K, dimensionality) means for each distribution
%       sigma = (K, dimensionality, dimensionality) Covariance matrices
%       pi = (K, 1) prior class probabilities
%       points (n_points, dimensionality)
%
%       post will have shape (K, n_points)

        K = size(mu, 1);
        n_examples = size(points, 1);
        post = zeros(n_examples, K);
        
        % Compute likelihood for each point
        L = zeros(K, n_examples);
        for k = 1:K
            L_cur = mvnpdf(points, mu(k, :), squeeze(sigma(k, :, :)));
            % Clip small likelihood values
            L_cur(L_cur == 0) = 1e-7;
            L(k, :) = L_cur;
        end
        
        % Compute posterior using Bayes rule
        for i = 1:n_examples
            for k = 1:K
               post(i, k) = pi(k) * L(k, i);
               if thresholding == false
                   post(i, k) = post(i, k) / dot(pi, L(:, i));
               end
            end
        end
end