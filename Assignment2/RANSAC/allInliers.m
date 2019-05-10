function [maxInliers, maxCount] = allInliers(matches, threshold)
    maxCount = 0;
    maxInliers = true(1, size(matches,2));

    for i = 1:100
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