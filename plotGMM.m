function plotGMM(mu, sigma, train_images, n_plot_imgs, fig_title)
% Plot GMM ellipsoids with some data points
%
% Displays 4 views of the 3D data.
%
% plotGMM(mu, sigma, train_images, n_plot_images, fig_title)
%   (mu, sigma) = parameterize GMM
%   train_images = cell array of training data images, with shape
%       (n_images, 1)
%   n_plot_imgs = how many images to sample points from in the figure
%   fig_title = title to include in each subplot


    n_train = size(train_images, 1);
    plot_idxs = datasample([1:n_train], n_plot_imgs, 'Replace', false);
    vis_points = vectorize_images(train_images, plot_idxs);

    % Check if we only have one gaussian
    if length(size(sigma)) == 2
        sigma = reshape(sigma, 1, size(sigma, 1), size(sigma, 2));
    end
    
    quick_scatter = @(X) scatter3(X(:, 1), X(:, 2), X(:, 3), 20, X./255, '.');

    fig = figure();
    % Set params for viewing 3D plots 
    cameras = [5, 5;
               -45, 80;
               45, 80;
               45, 160];
    n_cameras = size(cameras, 1);
    plot_num = 0;
    for img_num = 1 : n_cameras
        plot_num = plot_num + 1;
        subplot(2, 2, plot_num);
        hold on
        % Plot the data
        quick_scatter(vis_points);
        % Plot the GMM Ellipsoids
        for k = 1 : size(mu, 1)
            [x, y, z] = ellipsoid(mu(k, 1), mu(k, 2), mu(k, 3), ...
                                  sqrt(sigma(k, 1, 1)), ...
                                  sqrt(sigma(k, 2, 2)), ...
                                  sqrt(sigma(k, 3, 3)));
            % Use to color the ellipsoid
            normalized_mu = mu(k, :) / norm(mu(k, :));
            s = surf(x, y, z, 'FaceAlpha', 0.8, 'FaceColor', normalized_mu);
                                  
        end
        view(cameras(img_num, 1), cameras(img_num, 2));
        title(sprintf('%s-camera=[%d,%d]', fig_title, cameras(img_num, 1), cameras(img_num, 2)));
    end


end