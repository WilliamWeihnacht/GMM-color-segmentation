function [predictions, areas] = measureDepth(mu, sigma, pi, test_images, img_names, output_dir, tau, B)
% Estimates distance of each test image from the camera.
% Results are saved in a table in save_name
%
% Assumes B is computed using distanceModel() so that B = [b, m]'
% to model d = b + m/(x^2)
%
%   predictions = measureDepth(mu, sigma, pi, test_images, test_names, B, save_name)
%       (mu, sigma, pi) = parameterize a GMM
%       test_images = cell array of images, with shape (n_images, 1)
%       test_names = cell array of names for the test_images with shape
%           (n_images, 1)
%       B = Distance model parameters [b, m]' so d=b+(m/x^2))
%       save_name = string filename to save table to
%       

    segmented_images = testGMM(mu, sigma, pi, tau, test_images);
    
    n_images = size(test_images, 1);
    
    predictions = zeros(n_images, 1);
    areas = zeros(n_images, 1);
    
    for img_num = 1 : n_images
       [A, center, radius] = ball_area(segmented_images{img_num});
       depth = [1, (1 / (A^2))]*B;
       predictions(img_num) = depth;
       areas(img_num) = A;
       % Show images
       fig = figure(1);
       subplot(1, 2, 1);
       imshow(test_images{img_num});
       title(sprintf('%s.jpg depth=%f', img_names{img_num}, depth));
       
       subplot(1, 2, 2);
       hold on
       imshow(segmented_images{img_num});
       viscircles(center, radius);
       title(sprintf('%s-depth=%f', img_names{img_num}, depth));
       hold off
       saveas(gcf, sprintf('%s/%s.png', output_dir, img_names{img_num}));
    end

end