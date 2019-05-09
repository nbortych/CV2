keep_columns = false(1,size(point_view_generated,2));
binary_pvm = point_view_generated>0;
for ii = 1:size(point_view_generated,2)
    if sum(binary_pvm(:,ii)) > 60
        keep_columns(ii) = true;
    end
end
point_view_generated = point_view_generated(: ,keep_columns);


vpm = point_view_generated;
binary = vpm>0;
binary(2:2:end,:) = [];
[r, c] = size(binary);                          
imagesc((1:c)+0.5, (1:r)+0.5, binary); 
colormap(gray);                             
