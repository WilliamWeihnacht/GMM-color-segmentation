function [A, center, radius] = ball_area(segmented_img)
% Computes the area of the ball in the segmented_img
% 
% area = ball_area(segmented_img)

        stats = regionprops('table', segmented_img, 'Centroid', ...
                             'MajorAxisLength', 'MinorAxisLength');
        diameters = mean([stats.MajorAxisLength stats.MinorAxisLength], 2);
        [ball_diameter, max_idx] = max(diameters);
        A = 3.1415926*((ball_diameter / 2)^2);
        center = stats.Centroid(max_idx, :);
        radius = ball_diameter / 2;
end

