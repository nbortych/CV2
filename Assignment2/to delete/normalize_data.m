function [p_norm, T] = normalize_data(p)
    N = size(p, 1);
    p_sum = sum(p, 1);
    mx = p_sum(1)/N;
    my = p_sum(2)/N;
    m = [mx, my];
    
    d = 0;
    for i = 1:1:size(p,1)
       d = d + norm(p(i,1:2)-m);
    end
    d = d/N;
    
    T = [sqrt(2)/d, 0, -mx*sqrt(2)/d; 0, sqrt(2)/d, -my*sqrt(2)/d; 0, 0, 1];
    p_norm = T*p';
%     disp(size(p));
%     disp(size(T));
    p_norm = p_norm';
    
    
    %Check and show that normalized data satisfy mean=0 and dev=sqrt(2)
    N = size(p_norm, 1);
    p_sum = sum(p_norm, 1);
    mx = p_sum(1)/N;
    my = p_sum(2)/N;
    m = [mx, my];
    
    d = 0;
    for i = 1:1:size(p_norm,1)
       d = d + norm(p_norm(i,1:2)-m);
    end
    d = d/N;
    
%     disp(["Normalized data has mean on x and y:", mx, my]);
%     disp(["Normalized data has average distance to the mean:", d]);
    
end
