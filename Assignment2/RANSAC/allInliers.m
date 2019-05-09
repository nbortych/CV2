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
