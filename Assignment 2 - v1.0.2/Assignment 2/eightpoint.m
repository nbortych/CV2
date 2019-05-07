path = 'Data/House';

im1 = im2single(imread(strcat(path, '/', + 'frame00000044.png')));
im2 = im2single(imread(strcat(path, '/', + 'frame00000045.png')));

matches = keypoint_matching(im1, im2);

[matches, T1, T2] = normalize_points(matches);

%F = fundamentalMatrix(matches);

%if normalize_bool == 1
%    F = T2 * F * T1;
%end

[all_inliers, count] = allInliers(matches, 0.00001);
what = matches(:, all_inliers);

%F = fundamentalMatrixRANSAC(matches, 0.00001);

function F = fundamentalMatrix(matches)
    x1 = matches(1,:)';
    y1 = matches(2,:)';
    x2 = matches(3,:)';
    y2 = matches(4,:)';
    o = ones(size(matches(1,:)));

    A = [x1.*x2, x1.*y2, x1, y1.*x2, y1.*y2, y1, x2, y2, o'];

    [U, D, V] = svd(A);

    F = V(:,end);
    F = reshape(F, [3,3])';

    [Uf, Df, Vf] = svd(F);
    Df(end:end) = 0;

    F = Uf * Df * Vf';
end

function F = fundamentalMatrixRANSAC(matches, threshold)
    maxCount = 0;
    maxInliers = true(1, size(matches,2));

    for i =1:100
        sampled_points = subSample(matches, 8);

        F = fundamentalMatrix(sampled_points);

        di = sampson_distance(sampled_points, F);
        inliers = di <= threshold;
        count = sum(inliers);
        
        if count > maxCount
            maxCount = count;
            maxInliers = inliers;
        end
    end
    
    F = fundamentalMatrix(matches(:, maxInliers));
end

function [maxInliers, maxCount] = allInliers(matches, threshold)
    maxCount = 0;
    maxInliers = true(1, size(matches,2));

    for i =1:100
        F = fundamentalMatrix(matches);

        di = sampson_distance(matches, F);
        inliers = di <= threshold;
        count = sum(inliers);
        
        if count > maxCount
            maxCount = count;
            maxInliers = inliers;
        end
    end
end

function [normalized_points, T1, T2] = normalize_points(matches)
    x1 = matches(1,:);
    y1 = matches(2,:);
    x2 = matches(3,:);
    y2 = matches(4,:);

    Mx1 = mean(x1);
    My1 = mean(y1);
    Mx2 = mean(x2);
    My2 = mean(y2);
    
    d1 = mean(sqrt((x1 - Mx1).^2 + (y1 - My1).^2));
    d2 = mean(sqrt((x2 - Mx2).^2 + (y2 - My2).^2));
    
    T1 = [sqrt(2)/d1, 0, -Mx1*sqrt(2)/d1; 0, sqrt(2)/d1, -My1*sqrt(2)/d1; 0, 0, 1];
    T2 = [sqrt(2)/d2, 0, -Mx2*sqrt(2)/d2; 0, sqrt(2)/d2, -My2*sqrt(2)/d2; 0, 0, 1];
        
    o = ones([1, size(x1, 2)]);
    
    matches1 = [x1; y1; o];
    matches1 = T1 * matches1;
    
    matches2 = [x2; y2; o];
    matches2 = T2 * matches2;
    
    normalized_points = cat(1,matches1,matches2);
    
    normalized_points(6,:) = [];
    normalized_points(3,:) = [];
end

function sampled_points = subSample(matches, N)
    x1 = matches(1,:);
    y1 = matches(2,:);
    x2 = matches(3,:);
    y2 = matches(4,:);

    selection = randperm(size(x1, 2), N);
    x1 = x1(:,selection);
    y1 = y1(:,selection);
    x2 = x2(:,selection);
    y2 = y2(:,selection);
    
    sampled_points = cat(1,x1,y1,x2,y2);
end

function di = sampson_distance(sampled_points, F)
    x1 = sampled_points(1,:);
    y1 = sampled_points(2,:);
    x2 = sampled_points(3,:);
    y2 = sampled_points(4,:);
    
    o = ones([1, size(x1, 2)]);

    matches1 = [x1; y1; o];
    matches2 = [x2; y2; o];
    
    di = diag(((matches2' * F * matches1).^2)./(sum((F*matches1).^2) + sum((F*matches2).^2)));
end
