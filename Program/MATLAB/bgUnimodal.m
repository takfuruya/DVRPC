%{
Covariance is positive definite.
1.mp4 (1hr --- 72001frames)
LAB
- 2590.362045 seconds.
- Calculating mean: 822.711387 seconds.
- Calculating covariance: 1297.251297 seconds.
RGB
- 1895.093065 seconds.
- Calculating mean: 212.910855 seconds.
- Calculating covariance: 674.357679 seconds.
Requires:
    - topdm.m
%}

close all;
clearvars -except vidObj;
clc;

%frameRange = 1:1000;
frameRange = randperm(vidObj.NumberOfFrames, 1000);

nFrames   = numel(frameRange);
vidHeight = vidObj.Height;
vidWidth  = vidObj.Width;
nPixels   = vidHeight * vidWidth;
vidSize   = [nFrames nPixels];
rgb2lab   = makecform('srgb2lab');

tic;

% Separate video into channels
vidR = zeros(vidSize);
vidG = zeros(vidSize);
vidB = zeros(vidSize);

for f = 1 : nFrames
    disp([num2str(f) ' frame: ' num2str(frameRange(f))]);
    im = read(vidObj, frameRange(f));
    %im = applycform(im, rgb2lab);
    imR = im(:, :, 1);
    imG = im(:, :, 2);
    imB = im(:, :, 3);
    vidR(f, :) = imR(:);
    vidG(f, :) = imG(:);
    vidB(f, :) = imB(:);
end

toc;
tic;

% Get mean
disp('Calculating mean...');

imMeanR = mean(double(vidR));
imMeanG = mean(double(vidG));
imMeanB = mean(double(vidB));

imMean = [imMeanR(:) imMeanG(:) imMeanB(:)];

toc;
tic;

% Get covariance
disp('Calculating covariance...');

imCov = zeros(3, 3, nPixels);
imInvCov = zeros(3, 3, nPixels);
for i = 1 : nPixels
    c = cov([vidR(:, i) vidG(:, i) vidB(:, i)]);
    
    % Make sure covariance is positive definite
    c = topdm(c);
    
    % Would improve if ill conditioned case was taken into account
    imCov(:, :, i) = c;
    imInvCov(:, :, i) = inv(c);
end

toc;


% subplot(2, 2, 1);
% imshow(uint8(imMean));
% title('Mean');
% subplot(2, 2, 2);
% imagesc(imCovR);
% title('Variance R');
% subplot(2, 2, 3);
% imagesc(imCovG);
% title('Variance G');
% subplot(2, 2, 4);
% imagesc(imCovB);
% title('Variance B');

%save('../Data/BG_unimodal_vid1_random1000.mat', 'imCov', 'imMean', 'imInvCov')
