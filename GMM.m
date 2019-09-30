%% GMM for Color Segmention


%% Load Training Data

files = dir('train_images/*.jpg');
n_train = length(files);
train_images = cell(n_train, 1);
distances = zeros(n_train, 1);

for img_num = 1 : n_train
    % Load each image
    img = imread(sprintf('train_images/%s', files(img_num).name));
    train_images{img_num} = img;
    distances(img_num) = str2num(strrep(files(img_num).name, '.jpg', ''));
end

%% Preprocess the data

% Combine all pixels into a single vector
raw_img_vec = vectorize_images(train_images, [1:n_train]);

% Extract orange ball pixels to use as training data
training_vec = [];
for img_num = 1 : n_train
    current_img = train_images{img_num};
    % Load masks
    name_num = strrep(files(img_num).name, '.jpg', '');
    load(sprintf('masks/%s/%sx.mat', name_num, name_num));
    load(sprintf('masks/%s/%sy.mat', name_num, name_num));
    % Get mask for orange pixels
    x = floor(x);
    y = floor(y);
    mask = roipoly(current_img, x, y);
    mask_3dim = repmat(mask, 1, 1, 3);
    % Extract orange pixels
    current_training = current_img(mask_3dim);
    current_training = reshape(current_training, size(current_training, 1) / 3, 3);
    
    training_vec = vertcat(training_vec, current_training);
end

training_vec = double(training_vec);

%% Load Test Data

test_files = dir('test_images/*.jpg');
n_test = length(test_files);
test_images = cell(n_test, 1);
test_names = cell(n_test, 1);

for img_num = 1 : n_test
    % Load each image
    img = imread(sprintf('test_images/%s', test_files(img_num).name));
    test_images{img_num} = img;
    test_names{img_num} = strrep(test_files(img_num).name, '.jpg', '');
end


%% Single Gaussian

mu = mean(training_vec);
sigma = cov(training_vec);
% pi acts as a scaling factor in the single gaussian case
% that makes us need a larger tau, so leave pi as 1
pi = 1;

tau = 1e-6;

% Estimate distance using single gaussian segmentations
B_single = distanceModel(mu, sigma, pi, train_images, distances, tau);

mkdir results/single_gaussian
[predictions_single, areas_single] = measureDepth(mu, sigma, pi, test_images, ...
                                                  test_names, 'results/single_gaussian', ....
                                                  tau, B_single);
% Visualize gaussian ellipsoid
plotGMM(mu, sigma, train_images, 5, 'SingleGaussian');

% Set figure size
set(gcf, 'PaperUnits', 'inches');
x_width=9.125 ;y_width=7.25;
set(gcf, 'PaperPosition', [0 0 x_width y_width]); 
saveas(gcf, 'single_gaussian.png');

save('results/single_gaussian.mat', 'mu', 'sigma', 'pi', 'tau', 'B_single');

%% GMM Training

K = 3;
[mu, sigma, pi] = initialize_GMM(K, 3, max(raw_img_vec), min(raw_img_vec));
% Initialize first distribution using orange pixels
mu(1, :) = mean(training_vec);
sigma(1, :, :) = cov(training_vec);

% Select subset of all pixels to use for non-orange GMM pixels
data_idxs = datasample([1 : size(raw_img_vec, 1)], 50000, 'Replace', false); 
training_data = raw_img_vec(data_idxs, :);
training_data = vertcat(training_vec, training_data);

[mu, sigma, pi] = trainGMM(mu, sigma, pi, training_data, K, 50, 0.1); 

%% GMM Application

tau = 1e-7;

% Estimate distance using GMM segmentations
mkdir results/gmm
B_gmm = distanceModel(mu, sigma, pi, train_images, distances, tau);
[predictions_gmm, areas_gmm] = measureDepth(mu, sigma, pi, test_images, ...
                                            test_names, 'results/gmm', ...
                                            tau, B_gmm);

% Visualize gaussian ellipsoids
plotGMM(mu, sigma, train_images, 5, sprintf('GMM K=%d', K));

% Set figure size
set(gcf, 'PaperUnits', 'inches');
x_width=9.125;
y_width=7.25;
set(gcf, 'PaperPosition', [0 0 x_width y_width]); 
saveas(gcf, 'GMM.png');


save('results/gmm.mat', 'mu', 'sigma', 'pi', 'tau', 'B_gmm');

%% Combine distance predictions

predictions = table(test_names, predictions_single, predictions_gmm, ...
                    'VariableNames', {'test_img', 'single_gaussian', 'gmm'});
save('results/predictions.mat', 'predictions');