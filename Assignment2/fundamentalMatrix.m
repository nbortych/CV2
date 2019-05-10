function F = fundamentalMatrix(matches)
    x1 = matches(1,:)';
    y1 = matches(2,:)';
    x2 = matches(3,:)';
    y2 = matches(4,:)';
    o = ones(size(matches(1,:)));

    A = [x1.*x2, x1.*y2, x1, y1.*x2, y1.*y2, y1, x2, y2, o'];

    [U_a, D_a, V_a] = svd(A);

    F = V_a(:,end);
    F = reshape(F, [3,3])';

    [U_f, D_f, V_f] = svd(F);
    D_f(end:end) = 0;

    F = U_f * D_f * V_f';
end