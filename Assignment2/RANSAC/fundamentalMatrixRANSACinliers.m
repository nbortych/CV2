function [F, maxInliers] = fundamentalMatrixRANSACinliers(matches, threshold)
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