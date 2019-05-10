path = 'Data/House';

im1 = im2single(imread(strcat(path, '/', + 'frame00000044.png')));
im2 = im2single(imread(strcat(path, '/', + 'frame00000045.png')));

matches = keypoint_matching(im1, im2);

F = fundamentalMatrix(matches);
figure(1);
imshow(im2);
hold on;    
drawEpipolarLines(F, im2);

[matches, T1, T2] = normalize_points(matches);
F = fundamentalMatrix(matches);
F = T2' * F * T1;
figure(2);
imshow(im2);
hold on;  
drawEpipolarLines(F, im2);

[matches, T1, T2] = normalize_points(matches);
F = fundamentalMatrixRANSAC(matches, 0.0004);
F = T2' * F * T1;
figure(3);
imshow(im2);
hold on;  
drawEpipolarLines(F, im2);