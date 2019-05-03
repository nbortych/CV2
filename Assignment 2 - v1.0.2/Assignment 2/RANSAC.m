function transform = RANSAC(image1, image2, print)
    matches = keypoint_matching(image1, image2, 0);

    gold_t = matches(3:4,:);
    P = 3;
    N = 7;
    if N > size(matches, 1)
        N = size(matches,1);
    end
    
    best_tform = [0 0 0 0 0 0]';
    best_score = 0;

    for run=1:N
        score = 0;
        perm = randperm(size(matches,2));
        sel = perm(1:P);
        m_sel = matches(:,sel);
        transform = m_sel(3:4,:);
        b = reshape(transform,[P*2,1]);
        A = get_A(m_sel(1:2,:))
        
        tform = pinv(A)*b;
        
        m = [tform(1) tform(2); tform(3) tform(4)];
        t = [tform(5), tform(6)]';
        
        for i = 1:size(gold_t,2)
            xy = [matches(1,i) matches(2,i)]';
            uv = (m * xy) + t;
            d = norm(gold_t(:,i) - uv);
            if d < 10
                score = score + 1;
            end
        end
        
        if score > best_score
            best_score = score;
            best_tform = tform;
        end
    end
    
    transform = best_tform;
end

function A = get_A(points)
    A = [];
    for i = 1:size(points,2)
        x = points(1,i);
        y = points(2,i);
        A(end+1:end+2,:) = [x y 0 0 1 0 ; 0 0 x y 0 1];
    end
end