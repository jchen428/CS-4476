% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Gaussian filters
gaussian1 = fspecial('Gaussian', feature_width, 1);
gaussian2 = fspecial('Gaussian', 15, 1);

% Image gradient
[Gx, Gy] = imgradientxy(gaussian1);

% Image derivatives
Ix = imfilter(image, Gx);
Ixx = imfilter(Ix .* Ix, gaussian2);
Iy = imfilter(image, Gy);
Iyy = imfilter(Iy .* Iy, gaussian2);
Ixy = imfilter(Ix .* Iy, gaussian2);

% Cornerness function
alpha = 0.05;
harris = Ixx .* Iyy  - Ixy .^ 2 - alpha .* (Ixx + Iyy) .^ 2;

% Thresh cornerness values with logical matrix
threshold = harris > 0.0001 * max(harris(:));
harris = harris .* threshold;

% Sliding window
maxima = colfilt(harris, [feature_width feature_width], 'sliding', @max);
harris = harris .* (harris == maxima);

% Exclude values near edges
harris(1 : feature_width, :) = 0;
harris(: , 1 : feature_width) = 0;
harris(: , end - feature_width : end) = 0;
harris(end - feature_width : end, :) = 0;

% Find all nonzero values
[y, x] = find(harris);

end
