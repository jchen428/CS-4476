% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
% that you wrote for part II.

N = 3000;
numPoints = size(matches_a, 1);

% Iteratively RANSAC for best fundamental matrix
maxConfidence = 0;
Best_Fmatrix = [];
for i = 1 : N
	samples = randsample(numPoints, 8);
	sampleA = matches_a(samples, : );
	sampleB = matches_b(samples, : );
    
	currF = estimate_fundamental_matrix(sampleA, sampleB);
   
	% Find inliers
	currInliersA = zeros(numPoints, 2);
	currInliersB = zeros(numPoints, 2);
    distThreshold = 0.0001;
    
    for j = 1 : numPoints
        dist = abs([matches_b(j, : ), 1] * currF * [matches_a(j, : ), 1]');
        if dist < distThreshold
            currInliersA(j, : ) = matches_a(j, : );
            currInliersB(j, : ) = matches_b(j, : );
        end
    end
    currInliersA(~any(currInliersA, 2), : ) = [];     % Remove zero rows
    currInliersB(~any(currInliersB, 2), : ) = [];
    
    % Rate confidence and save best inliers and fundamental matrix
    numInliers = size(currInliersA, 1);
    confidence = numInliers / numPoints;
    
    if confidence > maxConfidence
        maxConfidence = confidence;
        Best_Fmatrix = currF;
        inliers_a = currInliersA;
        inliers_b = currInliersB;
    end
end

% % Only show 30 random inliers
% indices = randsample(size(inliers_a, 1), 30);
% inliers_a = inliers_a(indices, : );
% inliers_b = inliers_b(indices, : );

end

