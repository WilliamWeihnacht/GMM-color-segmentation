function img_vec = vectorize_images(img_cell, idxs)
% Helper function to extract images with given indexes from img_cell
% into a single vector
%
% Resulting shape of img_vec is (n_pixels, n_channels)
%
% Assumes images are uint8 data in channels-last order
% Assumes images have same number of channels
% 
%   img_vec = extract_images(idxs, img_struct)

    % Preallocate img_vec
    n_pixels = 0;
    for i = 1 : length(idxs)
        img = img_cell{i};
        n_pixels = n_pixels + size(img, 1)*size(img, 2);
    end 
    img_vec = zeros(n_pixels, 3, 'uint8');
    
    % Unwrap images into img_vec
    next_index = 1;
    for i = 1 : length(idxs)
        img = img_cell{i};
        img_pixels = size(img, 1)*size(img, 2);
        img_vec(next_index : next_index + img_pixels - 1, :) = reshape(img, [img_pixels, size(img, 3)]);
        next_index = next_index + img_pixels;
    end
    % Convert to double so computation works
    img_vec = double(img_vec);
   
end

