% Projection Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu, Grady Williams, James Hays

% Returns the projection matrix for a given set of corresponding 2D and
% 3D points. 

% 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
% 'Points_3D' is nx3 matrix of 3D coordinate of points in the world

% 'M' is the 3x4 projection matrix


function M = calculate_projection_matrix(Points_2D, Points_3D)

% To solve for the projection matrix. You need to setup a homogenous
% set of equations using the corresponding 2D and 3D points:

%                                                     [M11       [ u1
%                                                      M12         v1
%                                                      M13         .
%                                                      M14         .
%[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
%  0  0  0  0 X1 Y1 Z1 1 -v1*Z1 -v1*Y1 -v1*Z1          M22         .
%  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
%  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
%  0  0  0  0 Xn Yn Zn 1 -vn*Zn -vn*Yn -vn*Zn ]        M31         .
%                                                      M32         un
%                                                      M33         vn ]

% Then you can solve this using least squares with the '\' operator or SVD.
% Notice you obtain 2 equations for each corresponding 2D and 3D point
% pair. To solve this, you need at least 6 point pairs.

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

%Your total residual should be less than 1.

numPoints = size(Points_2D, 1);
A = zeros(2 * numPoints, 12);

% Construct homogeneous linear system
for i = 1 : numPoints
	point2D = Points_2D(i, : ); 
	u = point2D(1);
	v = point2D(2);
	
	point3D = Points_3D(i, : );
	x = point3D(1);
	y = point3D(2);
	z = point3D(3);
	
	A(2 * i - 1 : 2 * i, : ) = [ x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u; ...
                                0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v ];
end

% Solve for projection matrix
[~, ~, V] = svd(A);
M = V( : , end);
M = reshape(M, [], 3)';

end

