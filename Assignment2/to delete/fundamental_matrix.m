function [F] = fundamental_matrix(p1, p2, normalize)

    %Do normalization if asked
    if normalize
        [p1, T1] = normalize_data(p1);
        [p2, T2] = normalize_data(p2);
    end

    % Generate A matrix and do SVD
    A = [];
    for i = 1:1:size(p1,1)
       line = p1(i,:)' * p2(i,:);

       line = reshape(line, [1,9]);
       A = [A; line];
    end
    
    
    [U,D,V] = svd(A); 
    

    %Find smaller singular value of D
    sv_vector = diag(D);
    [min_sv, min_idx] = min(sv_vector);

    %F matrix is the column of V corresponding to lowest singular value (should
    %be last column, but let's still check for it)
    F = V(:,min_idx);
    F = reshape(F, [3,3]);
    [Uf,Df,Vf] = svd(F);
    
    
    %Find smaller singular value of Df and set it to zero
    sv_vector_f = diag(Df);
    [min_sv_f, min_idx_f] = min(sv_vector_f);
    Df(min_idx_f,min_idx_f) = 0;


    %Recompute F with the new corrected matrix Df
    F = Uf*Df*Vf';
    
    
%     test = A*reshape(F, [9, 1]);
%     max(test)
%     mean(test)
    
    if normalize
        F = T2' * F * T1;
    end


end
