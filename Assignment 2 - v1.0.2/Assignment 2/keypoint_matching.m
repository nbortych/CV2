function keypoint_matches = keypoint_matching(im1, im2)
    [f1,d1] = vl_sift(im1);
    [f2,d2] = vl_sift(im2);
    [matches, scores] = vl_ubcmatch(d1, d2);
    
    matches1 = f1([1,2], matches(1,:));
    matches2 = f2([1,2], matches(2,:));
    keypoint_matches = cat(1,matches1,matches2);
end