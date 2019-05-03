% read in point_view_matrix
point_view_matrix = dlmread('PointViewMatrix.txt');
% Number of rows 
size(point_view_matrix)
data_dir = './Data/House/';
N=50;
for i=1:N
    if i<10
        image = strcat(data_dir ,'frame0000000', str(i), '.png');
    else
        image = strcat(data_dir ,'frame000000', str(i), '.png');
    end
end