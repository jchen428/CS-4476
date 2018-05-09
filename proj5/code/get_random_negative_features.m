% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
n = length(image_files);
d = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_neg = zeros(num_samples, d);
currSamples = 1;

while currSamples <= num_samples
    % Randomly sample images until you have enough samples
    i = randi(n);
    image = rgb2gray(imread(strcat(non_face_scn_path, '/', image_files(i).name)));

    % Randomly scale the image
    minSize = feature_params.template_size / min(size(image));
    image = imresize(image, minSize + rand * (1 - minSize));

    % Get a random sample from the selected image
    sampleR = randi(size(image, 1) - feature_params.template_size);
    sampleC = randi(size(image, 2) - feature_params.template_size);
    sample = image(sampleR : sampleR + feature_params.template_size - 1, sampleC : sampleC + feature_params.template_size - 1);

    % Get HOG representation
    hog = vl_hog(single(sample), feature_params.hog_cell_size);

    % Reshape and HoG and put in features_neg
    features_neg(currSamples, : ) = reshape(hog, 1, d);

    % Increment sample counter
    currSamples = currSamples + 1;
end

end
