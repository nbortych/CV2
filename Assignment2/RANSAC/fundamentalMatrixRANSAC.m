function F = fundamentalMatrixRANSAC(matches, threshold)
    maxCount = 0;
    inlierMask = true(1, size(matches,2));

    for i =1:100
        sampled_points = subSample(matches, 8);

        F = fundamentalMatrix(sampled_points);

        d = sampsonDistance(sampled_points, F);
        inliers = d <= threshold;
        count = sum(inliers);
        
        if count > maxCount
            maxCount = count;
           	inlierMask = inliers;
        end
    end
    F = fundamentalMatrix(matches(:, inlierMask));
end