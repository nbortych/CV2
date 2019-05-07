function d = sampsonDistance(sampled_points, F)
    x1 = sampled_points(1,:);
    y1 = sampled_points(2,:);
    x2 = sampled_points(3,:);
    y2 = sampled_points(4,:);
    
    o = ones([1, size(x1, 2)]);

    matches1 = [x1; y1; o];
    matches2 = [x2; y2; o];
    
    d = diag(((matches2' * F * matches1).^2)./(sum((F*matches1).^2) + sum((F*matches2).^2)));
end