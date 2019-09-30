function B = distanceModel(mu, sigma, pi, train_images, distances, tau)
% Model distance as a function of area
%
%   Returns B = [b, m]' , the LS solution to modeling distance using
%       b + m/(x^2)
%
%   B = distanceModel(mu, sigma, pi, train_images, distances, tau)
%       (mu, sigma, pi) parameterize a GMM
%       train_images = a cell array containing images to segment
%       distances = vector of distances for each image in train_images
%       tau = thresholding parameter used when segmenting train_images
    
    segmented_images = testGMM(mu, sigma, pi, tau, train_images);
    n_images = size(segmented_images, 1);
   
    % Get ball area for each image
    areas = zeros(n_images, 1);
    for img_num = 1:n_images
        segmented = segmented_images{img_num};
        areas(img_num) = ball_area(segmented);
    end
    
    % Model distance as b + m/(x^2) -> [1, (1/x^2)] [b, m]' = d
    A = ones(n_images, 2);
    A(:, 2) = 1 ./ (areas .^ 2);
    
    
    B = A \ distances;

end

