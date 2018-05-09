% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

threshold = -0.1;

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

templateCellWidth = feature_params.template_size / feature_params.hog_cell_size;
d = templateCellWidth^2 * 31;

for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    cur_confidences = zeros(0,1);
    cur_bboxes = zeros(0,4);
    cur_image_ids = cell(0,1);
    
    % Iterate at each scale starting from smallest scale that can fit template
    for j = feature_params.template_size / min(size(img)) : 0.1 : 1
        % Resize image
        scale = imresize(img, j);
        
        % Image dimensions in number of cells
        cellRows = floor(size(scale, 1) / feature_params.hog_cell_size);
        cellCols = floor(size(scale, 2) / feature_params.hog_cell_size);
        
        % Get HoG representation
        hog = vl_hog(scale, feature_params.hog_cell_size);
        
        % Cell bounds of HOG
        r = cellRows - templateCellWidth + 1;
        c = cellCols - templateCellWidth + 1;
        X = zeros(r * c, d);
        
        % Iterate over HOG cells with sliding window reshape features into each row of X
        for k = 1 : r
            for l = 1 : c
                window = hog(k : k + templateCellWidth - 1, l : l + templateCellWidth - 1, : );
                X((k - 1) * c + l, : ) = reshape(window, 1, d);
            end
        end
        
        % Compute confidence scores and find linear indices above threshold
        confidenceScores = X * w + b;
        index = find(confidenceScores > threshold);
        
        % Get confidences above threshold and compute row/col indices from linear indices
        detectionConfidences = confidenceScores(index);
        detectionRows = floor(index / c);
        detectionCols = mod(index, c);
        
        % Compute (x, y) coordinates of the bounding boxes
        xMin = feature_params.hog_cell_size * detectionCols + 1;
        yMin = feature_params.hog_cell_size * detectionRows + 1;
        xMax = feature_params.hog_cell_size * (detectionCols + templateCellWidth);
        yMax = feature_params.hog_cell_size * (detectionRows + templateCellWidth);
        
        % Scale the bounding box to the current scale
        bbox = [xMin, yMin, xMax, yMax] / j;
        
        % Repeat the current image ID for the number of detections found
        imageID = repmat({test_scenes(i).name}, size(index, 1), 1);
        
        % Append current confidences, bounding boxes, and image IDs to their respective lists
        cur_confidences = [cur_confidences; detectionConfidences];
        cur_bboxes = [cur_bboxes; bbox];
        cur_image_ids = [cur_image_ids; imageID];
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end

end
