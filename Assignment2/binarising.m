%creating a binary pvm matrix
pvm = point_view_generated;
binary = pvm>0;
binary(2:2:end,:) = [];
[r, c] = size(binary);                          
imagesc((1:c)+0.5, (1:r)+0.5, binary); 
colormap(gray);                             
