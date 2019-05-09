keep_columns = false(1,size(point_view_generated,2));
binary_pvm = point_view_generated>0;
for ii = 1:size(point_view_generated,2)
    if sum(binary_pvm(:,ii)) > 60
        keep_columns(ii) = true;
    end
end
point_view_generated = point_view_generated(: ,keep_columns);



point_view_generated(point_view_generated == 0) = nan;
pvm =   point_view_generated; % dlmread('PointViewMatrix.txt'); %
block_size = 3; % Task Experiment: 3 vs 4
point_cloud = structure_from_motion(pvm,block_size);