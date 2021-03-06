addpath('./RANSAC')
point_view_generated = [];
distance_check = true;
ransac_inliers = false;
ransac_improvement = false;
distance_threshold = 1.6;
tol =  0.00000000000001; % A very small value.
%for each pair of matches
ransac_threshold_inliers= 1;
row_tol = 15;
for i=1:size(keypoint_matches, 1)
   
   %load the match
   keypoint_match = cell2mat(keypoint_matches(i));
   if ransac_improvement
      %generate transformation matrix
      transform = improvement_ransac(keypoint_match, 0);
      m = [transform(1) transform(2); transform(3) transform(4)];
      t = [transform(5), transform(6)]';
   end
   if ransac_inliers
        %find inliers theough ransac
        inliers = allInliers(keypoint_match, ransac_threshold_inliers);
        keypoint_match = keypoint_match(:, inliers);
   end
   
   %for each match
   for point=1:size(keypoint_match,2)
        
        x1 = keypoint_match(1,point);
        y1 = keypoint_match(2,point);
        x2 = keypoint_match(3,point);
        y2 = keypoint_match(4,point);
        %filter out matches that are far away
        if distance_check
            distance = sqrt( (x1-x2)^2 + (y1-y2)^2);
            if distance>distance_threshold
                continue
            end
        end
        %if it's the first pair
        if i == 1
            point_view_generated(1, size(point_view_generated,2)+1) = x1;
            point_view_generated(2, size(point_view_generated,2)) = y1;
            point_view_generated(3, size(point_view_generated,2)) = x2;
            point_view_generated(4, size(point_view_generated,2)) = y2;
        
            
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
    if ransac_improvement
        %for each point pair
        for point=1:size(point_view_generated,2)
          x1 = point_view_generated(2*i-1,point);
          y1 = point_view_generated(2*i,point);
          x2 = point_view_generated(2*i+1,point);
          y2 = point_view_generated(2*i+2,point);
          %if second row is empty
          if x2 == 0
              
              key1 = [x1; y1];
              %transform
              transformed = (m * key1) + t;
              %add to matrix
              point_view_generated(2*i+1,point) = transformed(1);
              point_view_generated(2*i+2,point) = transformed(2);
              
              
          end
        end
    end
end


%filtering out rows
keep_columns = false(1,size(point_view_generated,2));
binary = point_view_generated>0;
for j = 1:size(point_view_generated,2)
    if sum(binary(:,j)) > row_tol
        keep_columns(j) = true;
    end
end
point_view_generated = point_view_generated(: ,keep_columns);


%generate binary map
%creating a binary pvm matrix
pvm = point_view_generated;
binary = pvm>0;
binary(2:2:end,:) = [];
[r, c] = size(binary);                          
imagesc((1:c)+0.5, (1:r)+0.5, binary); 
colormap(gray);                             
