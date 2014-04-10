function BW = imMahalDist(im, bg, thresh)

[h, w, ~] = size(im);

imReshaped = reshape(im, h*w, 3);
distance = mahalanobisDistance(double(imReshaped), bg.imMean, bg.imInvCov);
imDist = reshape(distance, h, w);
%imDist = medfilt2(imDist, [5 5]);
BW = imDist > thresh;
%figure;
%imshow(BW);
%figure;
se = strel('disk', 3);
BW = imopen(BW, se);
BW = imclose(BW, se);
%imshow(BW);


