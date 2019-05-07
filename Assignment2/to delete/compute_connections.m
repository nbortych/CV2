function [connections] = compute_connections(PVM)

[m, n] = size(PVM);
assert (mod(m, 2) == 0);
m = m / 2;


connections = false(m, n);

for i =1:n
    for j =1:m
        if ~any(isnan(PVM(2*(j-1)+1:2*(j-1)+2, i)))
            connections(j, i) = true;
        end
    end
end

figure();
imshow(-1*(connections -1),'InitialMagnification','fit');
title(sprintf('Number of feature points:%d',size(PVM,2)));
daspect([n,m,1]);
pause(0.5);

end