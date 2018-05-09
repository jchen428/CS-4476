% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Find and sort euclidian distances between features
dists = pdist2(features1, features2);
[dists, ind] = sort(dists, 2);

% Element-wise division between first and second distance columns
% (distance ratios between features1 and its nearest 2 neighbors in features2)
nndr = dists(: , 1) ./ dists(: , 2);

% Elements in nndr that are less than threshold
threshold = 0.75;
confidences = 1 ./ nndr(nndr < threshold);

% Indices in features1 - indices in nndr that are less than threshold
i = find(nndr < threshold);
% Indices in features2 - indices in ind that are less than threshold
j = ind(nndr < threshold);

matches = [i,j];

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);

end
