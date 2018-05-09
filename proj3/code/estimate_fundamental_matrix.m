% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a, Points_b)

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

numPoints = size(Points_a, 1);
A = zeros(numPoints, 9);

% Compute centroid and scale for each image
cA = sum(Points_a, 1) / numPoints;
sA = sqrt(2) / norm(cA);
cB = sum(Points_b, 1) / numPoints;
sB = sqrt(2) / norm(cB);

% Compute transform matrix for image A
scaleA = [  sA, 0 , 0;
            0 , sA, 0;
            0 , 0 , 1 ];
offsetA = [ 1 , 0 , -cA(1);
            0 , 1 , -cA(2);
            0 , 0 , 1 ];
transformA = scaleA * offsetA;

% Compute transform matrix for image B
scaleB = [  sB, 0 , 0;
            0 , sB, 0;
            0 , 0 , 1 ];
offsetB = [ 1 , 0 , -cB(1);
            0 , 1 , -cB(2);
            0 , 0 , 1 ];
transformB = scaleB * offsetB;

% Construct homogeneous linear system
for i = 1 : numPoints
    % Normalized points
	pointB = transformA * [Points_a(i, : )'; 1];
	pointA = transformB * [Points_b(i, : )'; 1];
    % % Non-normalized points
    % pointB = Points_a(i, : );
    % pointA = Points_b(i, : );

	uA = pointA(1);
	vA = pointA(2);
	uB = pointB(1);
	vB = pointB(2);
	
	A(i, : ) = [ uA * uB, uA * vB, uA, vA * uB, vA * vB, vA, uB, vB, 1 ];
end

% Solve for fundamental matrix
[~, ~, V] = svd(A);
F_matrix = V( : , end);
F_matrix = reshape(F_matrix, [3, 3])';

% Enforce rank 2 constraint
[U, S, V] = svd(F_matrix);
S(3, 3) = 0;
F_matrix = U * S * V';

% Normalize
F_matrix = transformB.' * F_matrix * transformA;

end

