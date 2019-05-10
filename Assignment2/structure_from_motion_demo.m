%3d Model
point_view_generated(point_view_generated == 0) = nan;
pvm =   point_view_generated; % dlmread('PointViewMatrix.txt'); %
block_size = 4; % Task Experiment: 3 vs 4
point_cloud = structure_from_motion(pvm,block_size);