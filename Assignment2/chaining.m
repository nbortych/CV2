% read in point_view_matrix
point_view_matrix = dlmread('PointViewMatrix.txt');
% Number of rows 
size(point_view_matrix);

point_view_generated = [];

tol =  0.01; % A very small value.
%for each pair of matches

for i=1:size(keypoint_matches)
   %load the match
   keypoint_match = cell2mat(keypoint_matches(i));
   %for each match
   for point=1:size(keypoint_match,2)
        x1 = keypoint_match(1,point);
        y1 = keypoint_match(2,point);
        x2 = keypoint_match(3,point);
        y2 = keypoint_match(4,point);
        %if it's the first pair
        if i == 1
            point_view_generated(1, size(point_view_generated,2)+1) = x1;
            point_view_generated(2, size(point_view_generated,2)) = y1;
            point_view_generated(3, size(point_view_generated,2)) = x2;
            point_view_generated(4, size(point_view_generated,2)) = y2;
        %elseif i== size(keypoint_match,2)
          
        else
        %check if the point in the first image is the same as in the
        %previous match pair
            
            is_point_x = ismembertol(x1, point_view_generated(2*(i)-1, :), tol);
            is_point_y = ismembertol(y1, point_view_generated(2*(i), :), tol);
            
            %if both coordinates are the same
            if and(is_point_x,is_point_y)
                %find index of this point and set point_view_generated to
                %that index
                [ii,jj]=find(and(abs(point_view_generated(2*(i)-1,:)-x1)<tol, abs(point_view_generated(2*(i),:)-y1)<tol));
                %point_view_generated(2*i-1, jj) = x1;
                %point_view_generated(2*i,jj) = y1;
                point_view_generated(2*i+1,jj) = x2;
                point_view_generated(2*i+2, jj) = y2;
            else
                point_view_generated(2*i-1, size(point_view_generated,2)+1) = x1;
                point_view_generated(2*i, size(point_view_generated,2)) = y1;
                point_view_generated(2*i+1, size(point_view_generated,2)) = x2;
                point_view_generated(2*i+2, size(point_view_generated,2)) = y2;
            end
        end

    end
end