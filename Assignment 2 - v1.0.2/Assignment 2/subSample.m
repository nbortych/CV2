function sampled_points = subSample(matches, N)
    x1 = matches(1,:);
    y1 = matches(2,:);
    x2 = matches(3,:);
    y2 = matches(4,:);

    permutation = randperm(size(x1, 2), N);
    x1 = x1(:,permutation);
    y1 = y1(:,permutation);
    x2 = x2(:,permutation);
    y2 = y2(:,permutation);
    
    sampled_points = cat(1,x1,y1,x2,y2);
end