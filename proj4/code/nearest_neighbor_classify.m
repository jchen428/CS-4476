% Starter code prepared by James Hays for Computer Vision

%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you can vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).

function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
   This can tell you which indices in train_labels match a particular
   category. Not necessary for simple one nearest neighbor classifier.

 D = vl_alldist2(X,Y) 
    http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator ' 
   vl_alldist2 supports different distance metrics which can influence
   performance significantly. The default distance, L2, is fine for images.
   CHI2 tends to work well for histograms.
 
  [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
  [Y,I] = SORT(X) if you're going to be reasoning about many nearest
  neighbors 

%}

% Calculate distances
dists = vl_alldist2(train_image_feats', test_image_feats');

% Sort them
[~, index] = sort(dists, 2);

% Pick the nearest neighbor
predicted_categories = train_labels(index( : , 1));

%%% Attempts at NBNN for extra credit didn't work out.
% %n = size(test_image_feats, 1);
% numDescriptors = size(test_image_feats, 2);
% classes = unique(train_labels);
% numClasses = size(classes, 1);

% for a = 1 : size(test_image_feats, 1)
%     totals = zeros(numClasses, 1);
%     for i = 1 : numDescriptors
%         for j = 1 : numClasses
%             distToTrainFeats = dists(strcmp(classes(j), train_labels), a); % distance to each feature of each image of class
%             %[~, index] = sort(trainFeatsOfClass, 1);
%             %[sortedDists, index] = sort(dists, 2);
%             totalDist = sum(distToTrainFeats);
%             
%             %totals(j) = totals(j) + (test_image_feats(a, i) - train_image_feats(index(a, 1), i))^2;
%             totals(j) = totals(j) + totalDist;
%         end
%     end
%     [~, nn] = min(totals);
%     predicted_categories(a) = classes(nn);
% end

% for a = 1 : size(test_image_feats, 1)
%     totals = zeros(numClasses, 1);
%     for i = 1 : numDescriptors
%         for j = 1 : numClasses
%             trainFeatsOfClass = train_image_feats(strcmp(classes(j), train_labels), : );
%             dists = abs(trainFeatsOfClass( : , i) - test_image_feats(a, i));
%             %[sortedDists, index] = sort(dists, 2);
%             minDist = min(dists);
%             
%             %totals(j) = totals(j) + (test_image_feats(a, i) - train_image_feats(index(a, 1), i))^2;
%             totals(j) = totals(j) + minDist^2;
%         end
%     end
%     [~, nn] = min(totals);
%     predicted_categories(a) = classes(nn);
% end

% centers = zeros(numClasses, numDescriptors);
% for i = 1 : numClasses
%     trainFeatsOfClass = train_image_feats(strcmp(classes(i), train_labels), : );
%     centers(i, : ) = mean(trainFeatsOfClass, 1);
% end
% % dists = vl_alldist2(centers', test_image_feats');
% % [~, index] = sort(dists, 1);
% % predicted_categories = classes(index(1, : ));
% 
% % for a = 1 : size(test_image_feats, 1)
% %     totals = zeros(numClasses, 1);
% %     for i = 1 : numClasses
% %         dists = abs(centers(i, : ) - test_image_feats(a, : ));
% %         totals(i) = totals(i) + norm(dists);
% %     end
% %     [~, nn] = min(totals);
% %     predicted_categories(a) = classes(nn);
% % end
% for a = 1 : size(test_image_feats, 1)
%     totals = zeros(numClasses, 1);
%     for i = 1 : numDescriptors
%         featDists = zeros(numDescriptors);
%         for j = 1 : numClasses
%             dists = abs(centers(j, i) - test_image_feats(a, i));
%             %totals(j) = totals(j) + norm(dists);
%         end
%         
%     end
%     [~, nn] = min(totals);
%     predicted_categories(a) = classes(nn);
% end

end
