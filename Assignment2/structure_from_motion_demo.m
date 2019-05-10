block_size = 4;

% Note: One of both tests needs to be commented out.

%%%%%%%%% Test for generated point view matrix:
load("42.mat");
point_view_generated(point_view_generated == 0) = nan;
point_cloud_generated = structure_from_motion(point_view_generated,block_size);
%%%%%%%%%

%%%%%%%%% Test for example point view matrix:
% pvm = dlmread('PointViewMatrix.txt');
% point_cloud_example= structure_from_motion(pvm,block_size);
%%%%%%%%%
