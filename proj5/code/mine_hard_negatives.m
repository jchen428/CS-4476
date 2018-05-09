function [ features_neg ] = mine_hard_negatives( non_face_scn_path, feature_params, ...
    bboxes, confidences, image_ids )
%HARD_NEGATIVE_MINE Summary of this function goes here
%   Detailed explanation goes here

n = size(confidences, 1);
d = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_neg = zeros(n, d);
bboxes = round(bboxes);

for i = 1 : n
    image = imread(strcat(non_face_scn_path, '/', image_ids{i}));
    
    bbox = bboxes(i, : );
    bbox(1) = max(bbox(1), 0);
    bbox(2) = max(bbox(2), 0);
    bbox(3) = min(bbox(3), size(image, 2));
    bbox(4) = min(bbox(4), size(image, 1));
        
    image = image(bbox(2) : bbox(4), bbox(1) : bbox(3));
    image = imresize(image, [feature_params.template_size, feature_params.template_size]);
    
    hog = vl_hog(single(image), feature_params.hog_cell_size);
    
    features_neg(i, : ) = reshape(hog, 1, d);
end

end
