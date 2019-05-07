clear all;
close all;
clc;

MIN_NUMBER_OF_ELEMENTS = 10;
SIFT_THRESHOLD = 5;
MATCHING_THRESHOLD = 5;

DISTANCE_THRESHOLD = 1.3;
USE_RANSAC= true;
USE_DISTANCE=true;

MERGE_WITH_LAST_VIEW = false;

BASE_PATH = './Data/House/';

% Parse images

names = dir(strcat(BASE_PATH, '*.png'));

N_VIEWS = length(names);

images = cell(N_VIEWS, 1);
for i = 1:length(names)
   images{i} = imread(strcat(BASE_PATH, names(i).name));
end



points = cell(N_VIEWS, 1);
for i = 1:size(images,1)
   [f, d] = vl_sift(single(images{i}), 'PeakThresh', SIFT_THRESHOLD);
   points{i}.coordinates = f(1:2, :);
   points{i}.descriptors = d;
end

if MERGE_WITH_LAST_VIEW
   points{N_VIEWS+1} = points{1};
end

%% CHAINING

max_rows = 2*N_VIEWS;
point_view = NaN(max_rows, size(points{1}.coordinates, 2));

point_view(1:2,:) = points{1}.coordinates;

indexes = 1:size(points{1}.coordinates, 2);

for i=2:length(points)
    % calculate current row inside of point_view matrix 
    % (each cell of this matrix has two vertical elements)
    current_row = (mod(i-1, N_VIEWS))*2+1;
%     current_row = (i-1)*2+1;
    
    N_POINTS = size(points{i}.coordinates, 2);
    
    [matches, scores] = vl_ubcmatch(points{i-1}.descriptors, points{i}.descriptors, MATCHING_THRESHOLD);
    
    initial_matches = size(matches, 2);
    
    if USE_DISTANCE && i <= N_VIEWS
        distances = sum((points{i}.coordinates(:, matches(2, :))' - points{i-1}.coordinates(:, matches(1, :))').^2, 2);
        
        inliers = distances < DISTANCE_THRESHOLD;
    
        fprintf('Inliers: %d/%d; preDist: %f, postDist: %f\n', sum(inliers), length(inliers), sum(distances), sum(distances(inliers)));
        matches = matches(:, inliers);
        scores = scores(:, inliers);
    end
   
    if USE_RANSAC
        p1 = points{i-1}.coordinates(:, matches(1, :))';
        p1(:, 3) = 1;
        p2 = points{i}.coordinates(:, matches(2, :))';
        p2(:, 3) = 1;
        
        [~, inliers, ~, ~, ~] = fundamental_matrix_RANSAC(p1, p2, true);
        matches = matches(:, inliers);
        scores = scores(:, inliers);
    end
    
    best_matches = NaN(1, N_POINTS);
    best_scores = 1000000*ones(1, N_POINTS);
    for t=1:size(matches, 2)
        if scores(t) < best_scores(matches(2, t))
           best_matches(matches(2, t)) = matches(1, t);
           best_scores(matches(2, t)) = scores(t);
        end
    end
    
    fprintf('Matches: %d/%d\n', sum(~isnan(best_matches)), initial_matches);
    
    points_set = ~isnan(best_matches);
    
    % filter and add points which were matched with previously added points    
    
    point_view(current_row:current_row+1, indexes(best_matches(points_set))) = points{i}.coordinates(:,points_set);
    
    
    new_indexes = NaN(1, N_POINTS);
    new_indexes(points_set) = indexes(best_matches(points_set));

    %*********************************************
    % filter and add new descritors to new columns
    %********************************************* 
    
    %enlarge descriptors and point-view matrix
    start_col = size(point_view,2)+1;

    point_view = [point_view, NaN(max_rows,sum(~points_set))];
    
    point_view(current_row:current_row+1,start_col:end) = points{i}.coordinates(:, ~points_set);
    new_indexes(~points_set) = start_col:size(point_view, 2);
    
    indexes = new_indexes;
   
end


%% DRAW MATRIX
[h,w] = size(point_view);

binary_mat = compute_connections(point_view);

keep_columns = false(1,w);
for ii = 1:w
    if sum(binary_mat(:,ii)) > MIN_NUMBER_OF_ELEMENTS
        keep_columns(ii) = true;
    end
end


D = point_view(: ,keep_columns);

dlmwrite('../pointview.mat', D);

compute_connections(D);