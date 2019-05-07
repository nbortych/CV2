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