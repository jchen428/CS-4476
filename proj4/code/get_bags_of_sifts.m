% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

load('vocab.mat')
vocab_size = size(vocab, 2);

n = size(image_paths, 1);
m = 1500; % 1500     m=20 in build_vocabulary and lambda=0.00005 in svm_classify
image_feats = zeros(n, vocab_size);

for i = 1 : n
    % Read image and get SIFT features
    image = imread(image_paths{i});
    [~, features] = vl_dsift(single(image), 'fast', 'step', 5);
    
    % vocab present - 1NN
    % m = 500  -> 48.9%, 138s
    % m = 1000 -> 50.8%, 239s
    % m = 1500 -> 51.5%, 341s
    % m = 2000 -> 51.9%, 438s
    % m = Inf  -> 52.5%, 522s
    % If a ton of features, randomly sample m features
    if size(features, 2) > m
        features = features( : , randsample(size(features, 2), m));
    end
    
    % Calculate distances and sort them
    dists = vl_alldist2(single(features), vocab);
    [~, index] = sort(dists, 2);
    
    % Bin features into nearest neighbor histogram of visual words
    histogram = zeros(vocab_size, 1);
    for j = 1 : size(index, 1)
        histogram(index(j, 1), 1) = histogram(index(j, 1), 1) + 1;
    end
    
    % Normalize histogram and save to image_feats
    histogram = histogram / norm(histogram);
    image_feats(i, : ) = histogram';
end

end
