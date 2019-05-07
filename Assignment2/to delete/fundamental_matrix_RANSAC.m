function [F, best_inliers, inliers_rateo, time_to_best, len] = fundamental_matrix_RANSAC(p1, p2, normalize)
    
    % the number of iterations
    N = 50;
    
    %number of matches to consider
    P = 8;
    
    %threshold for inliers
    threshold = 0.00001;

    len = size(p1, 1);
    
    best_inliers = false(len, 1);
    time_to_best = 0;
    
    
    for n = 1:1:N
        idx = randperm(len, min(P, len));
        
        F = fundamental_matrix(p1(idx, :), p2(idx, :), normalize);
        
        distances = sampson_distance(p1, p2, F);
        
        inliers = distances < threshold;
        
        if sum(inliers) > sum(best_inliers)
            best_inliers = inliers;
            time_to_best = n;
            %disp([sum(best_inliers), "inliers out of ", len, " pairs, at iter: ", n]);
            %fprintf("%f %% inliers, at iter: %d\n", 100*sum(best_inliers)/len, n);
        end
    end
    
    %finally, re-estimate the fundamental matrix using the largest set of
    %inliers found
    inliers_rateo = sum(best_inliers)/len;
%     disp({"At time ", time_to_best, " we have inliers rateo:", inliers_rateo});
%     disp(size(best_inliers));
    F = fundamental_matrix(p1(best_inliers, :), p2(best_inliers, :), normalize);
    
    
end



function [d] = sampson_distance(p1, p2, F)
    
    d = zeros(size(p1, 1), 1);
    
    for i = 1:size(p1, 1)
        
        t1 = F * p1(i, :)';
        t2 = F' * p2(i, :)';
        
        d(i) = (p2(i, :) * F * p1(i, :)')^2 / (t1'*t1 + t2'*t2);
        
    end
    

end