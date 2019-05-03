path = 'Data/House';
files = dir(strcat(path,'/*.png'));

L = length (files);
keypoint_matches = [];
for i=1:L-1
    im1 = im2single(imread(strcat(path, '/', + files(i).name)));
    im2 = im2single(imread(strcat(path, '/', + files(i+1).name)));
    matches = keypoint_matching(im1, im2);
    keypoint_matches = [keypoint_matches; {matches}];
end

