function [overall_points] = structure_from_motion(pv_matrix, block_size)
 
    m = size(pv_matrix,1)/2;
    n = size(pv_matrix,2);
    
    overall_points = NaN(n,3);

    % 1 : m-block_size
    for i=1:m-block_size+1

        % Select dense block from the point-view matrix
        block_points = pv_matrix((2*i-1):(2*i+2*block_size-2),:); 

        % Check which points are defined for all images of current block
        def_idx = all(~isnan(block_points)); % [1 x n]  binary

        % Check if any point of the defined points is new to the 
        % overall point collection 
        overall_def_idx = all(~isnan(overall_points'));
        new_idx = ~overall_def_idx(def_idx);


        % In case previously unknown points are included in the current block
        if sum(new_idx)>0

            % Construct 2M × N measurement matrix D
            D = block_points(:,def_idx);

            % Normalize the point coordinates by translating them to the mean 
            % of the points in each view in the dense block
            D = D - mean(D,2);

            % Apply SVD to D to express it as D = U ∗ W ∗ V 
            [~, W, V] = svd(D);
            W = W(1:3,1:3);
            V = V(:,1:3);

            % Calculate S (given in slides)
            S = sqrt(W)*V'; % 3 x num_points_in_block
            
            % In case no points collected so far 
            if sum(overall_def_idx)==0
                overall_points(def_idx,:) = S';

            % Otherwise:
            % Stitch current point set to main view using point correspondence
            else
                % For all points that are in the current block, but already have
                % a representation in overall_points:

                % (1) Get their coordinates in overall_points
                X = overall_points(overall_def_idx & def_idx,:);

                % (2) Get their current coordinates from S (calculated for this
                % block)
                Y = S(:, ~new_idx)';

                % (3) Get lin. transformation of the points in Y to 
                % best conform them to the points in overall_points 
                % (= main view)
                [~,~,transform] = procrustes(X, Y);

                % From matlab documentation
                % [https://nl.mathworks.com/help/stats/procrustes.html]
                % Note: cannot just add c because of dimension conflict
                c = transform.c(end,:);
                T = transform.T;
                b = transform.b;
                S_transformed = b*S'*T + c;

                % Add new points transformed to main view to overall_points
                new_points_transformed = S_transformed(new_idx,:);
                overall_points(def_idx & ~overall_def_idx,:) = new_points_transformed;
            end 
        end
    end
    
    % Remove all points that are not defined
    overall_points = overall_points(overall_def_idx,:);
    
    % Plot result
    figure(1);
    scatter3(overall_points(:,1),...
             overall_points(:,2),...
             -overall_points(:,3),...
             'MarkerEdgeColor','k',...
             'MarkerFaceColor','g');
     view(-30,10);
    
end
