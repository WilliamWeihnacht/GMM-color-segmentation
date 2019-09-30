function segmented_images = testGMM(mu, sigma, pi, tau, test_images)
% Perform color segmentation on test_images using the GMM
% Clustering is determined using tau a thresholding cutoff
%
% Assumes mu, sigma, pi are in the format output by 
% trainGMM(). See the documentation for trainGMM() for details.
%
% segmented_images = testGMM(mu, sigma, pi, tau, test_images)
%   See documentation for trainGMM() for the form of mu, sigma, pi
%   tau = Thresholding cutoff for cluster determination
%   test_images = Cell array of images, each of which is channels-last
    
    n_images = size(test_images, 1);
    
    % Check if we have only one Gaussian
    K = size(mu, 1);
    if length(size(sigma)) == 2
        sigma = reshape(sigma, 1, size(sigma, 1), size(sigma, 2));
    end
    
    segmented_images = cell(n_images, 1);
    for img_num = 1:n_images
        current_img = test_images{img_num};
        img_shape = size(current_img);
        
        % Compute posterior for each Gaussian
        img_vec = vectorize_images({current_img}, [1]);
        post = computePosterior(mu, sigma, pi, img_vec, true);
        % Reshape to image
        img_post = reshape(post, [img_shape(1), img_shape(2), K]);
        
        % Segmentation -- assumes 1st Gaussian is desired cluster
        segmented = img_post(:, :, 1) > tau;
        
        segmented_images{img_num} = squeeze(segmented);
    end

end