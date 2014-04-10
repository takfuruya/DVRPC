close all;
seed = rng;
isPlotting = false;


% Trajectories.
nTraj = 10000;
maxTrajDuration = 20*30; % 30 seconds.
trajStart = zeros(nTraj, 1); % Frame #, inclusive.
isUsed = false(nTraj, 1);
xHist = single(NaN(nTraj, maxTrajDuration));
yHist = xHist;
vxHist = xHist;
vyHist = xHist;
bHist = false(nTraj, maxTrajDuration);



% im.
im = read(vidObj, frameRange(1));
imGray = rgb2gray(im);
bw = imMahalDist(im, bg, 5); % Foreground.


% Points.
corners = detectHarrisFeatures(imGray);
nTracking = size(corners, 1);
points = corners.Location;
isValid = true(nTracking, 1);
isFG = getImageValuesAt(points, bw);


trackingIdx = (1:nTracking)';
trajStart(trackingIdx) = 1;
isUsed(trackingIdx) = true;
xHist(trackingIdx, 1) = points(:, 1);
yHist(trackingIdx, 1) = points(:, 2);
bHist(trackingIdx, 1) = isFG;


% KLT.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
initialize(pointTracker, points, imGray);


% Plotting.
if isPlotting
    hFigure = figure;
    set(hFigure, 'Position', [50 50 1000 800])
    hSubplot11 = subplot(2, 2, 1);
    hSubplot12 = subplot(2, 2, 2);
    hSubplot21 = subplot(2, 2, 3);
    hSubplot22 = subplot(2, 2, 4);
end


tic
for i = 2 : nFrames
    timeMessage = ['--- Frame: ' num2str(frameRange(i)) ', ' num2str(i) '/' num2str(nFrames) ' (' num2str(round(i/nFrames * 100)) '%) ---'];
    disp(timeMessage);
    
    im = read(vidObj, frameRange(i));
    imGray = rgb2gray(im);
    bw = imMahalDist(im, bg, 5);
    
    % Tracking
    prevPoints = points;
    prevIsValid = isValid;
    [points, isValid] = step(pointTracker, imGray);
    isValid = isValid & prevIsValid;
    velocity = points - prevPoints;
    
    
    % Filter points outside image.
    isXLargerThanWidth = points(:, 1) > vidWidth - 0.1;
    isYLargerThanHeight = points(:, 2) > vidHeight - 0.1;
    isValid = isValid & ...
              (points(:, 1) > 0.1) & (points(:, 2) > 0.1) & ...
              ~isXLargerThanWidth & ~isYLargerThanHeight;
    % Prevent error later but those values aren't used.
    points(points < 0.1) = 0.1;
    points(isXLargerThanWidth, 1) = vidWidth - 0.1;
    points(isYLargerThanHeight, 2) = vidHeight - 0.1;
    
    
    % Find if point is foreground.
    isFG = getImageValuesAt(points, bw);
    
    
    % Stop tracking background.
    % 1) Find points with trajectory of at least 5 frames.
    isValidAndLong = isValid & sum(~isnan(xHist(trackingIdx, :)), 2) > 3;
    isValidAndLongAndBG = false(nTracking, 1);
    % 2) Compute points that stopped moving in the background.
    idxValidAndLong = trackingIdx(isValidAndLong);
    base = i - trajStart(idxValidAndLong) + 1;
    idx1 = sub2ind([nTraj maxTrajDuration], idxValidAndLong, base - 1);
    idx2 = sub2ind([nTraj maxTrajDuration], idxValidAndLong, base - 2);
    idx3 = sub2ind([nTraj maxTrajDuration], idxValidAndLong, base - 3);
    b1 = ~isFG(isValidAndLong);
    b2 = ~bHist(idx1);
    b3 = ~bHist(idx2);
    b4 = ~bHist(idx3);
    k = 1.5^2;
    v1 = velocity(isValidAndLong, 1).^2 + velocity(isValidAndLong, 2).^2 < k;
    v2 = vxHist(idx1).^2 + vyHist(idx1).^2 < k;
    v3 = vxHist(idx2).^2 + vyHist(idx2).^2 < k;
    v4 = vxHist(idx3).^2 + vyHist(idx3).^2 < k;
    isValidAndLongAndBG(isValidAndLong) = b1 & b2 & b3 & b4 & v1 & v2 & v3 & v4;
    % 3) Set invalid and crop the last bit.
    for j = trackingIdx(isValidAndLongAndBG)
        temp = i - trajStart(j) - 2;
        xHist(j, temp:end) = NaN;
        yHist(j, temp:end) = NaN;
        vxHist(j, temp:end) = NaN;
        vyHist(j, temp:end) = NaN;
        bHist(j, temp:end) = false;
    end
    isValid(isValidAndLongAndBG) = false;
    
    
    % Store history
    idx = sub2ind([nTraj maxTrajDuration], trackingIdx(isValid), i-trajStart(trackingIdx(isValid))+1);
    xHist(idx) = points(isValid, 1);
    yHist(idx) = points(isValid, 2);
    vxHist(idx) = velocity(isValid, 1);
    vyHist(idx) = velocity(isValid, 2);
    bHist(idx) = isFG(isValid);
    
    
    % Remove short trajectories.
    idxInvalid = trackingIdx(~isValid);
    isShort = sum(~isnan(xHist(idxInvalid, :)), 2) < 5;
    idxShort = idxInvalid(isShort);
    isUsed(idxShort) = false;
    
    
    % Resample every 15 frames.
    if (mod(i, 15) == 0)
        corners = detectHarrisFeatures(imGray);
        corners = corners.Location;
        
        nValid = sum(isValid);
        nCorners = size(corners, 1);
        nTracking = nValid + nCorners;
        
        points = [points(isValid, :); corners];
        trackingIdx = [trackingIdx(isValid); find(~isUsed, nCorners)];
        
        idx = trackingIdx((nValid+1):end);
        trajStart(idx) = i;
        isUsed(idx) = true;
        xHist(idx, :) = NaN;
        yHist(idx, :) = NaN;
        vxHist(idx, :) = NaN;
        vyHist(idx, :) = NaN;
        bHist(idx, :) = false;
        
        isValid = true(nTracking, 1);
        setPoints(pointTracker, points, isValid);
    end
    
    % ===================== Plot ===========================
    nUsed = sum(isUsed);
    memoryMessage = ['Traj Total: ' num2str(nUsed) '/' num2str(nTraj) ' (' num2str(round(nUsed/nTraj*100)) '%) Active: ' num2str(nTracking)];
    disp(memoryMessage);
    
    if ~isPlotting
        continue
    end
    figure(hFigure);
    
    % KLT
    subplot(hSubplot11);
    imshow(im);
    title(timeMessage);
    
    %
    subplot(hSubplot12);
    imshow(im);
    hold on;
    plot(points(isValid, 1), points(isValid, 2), 'r.');
    plot(points(~isValid, 1), points(~isValid, 2), 'b.');
    labels = cellstr( num2str(trackingIdx) );
    text(double(points(:, 1)), double(points(:, 2)), labels, 'Clipping', 'on');
    hold off;
    title(memoryMessage);
    
    %
    subplot(hSubplot21);
    imshow(bw);
    hold on;
    plot(points(isValid, 1), points(isValid, 2), 'r.');
    plot(points(~isValid, 1), points(~isValid, 2), 'b.');
    hold off;
    
    subplot(hSubplot22);
    imshow(im);
    hold on;
    plot(xHist(trackingIdx(isValid), :)', yHist(trackingIdx(isValid), :)', 'r-');
    plot(points(isValid, 1), points(isValid, 2), 'g.', 'MarkerSize', 1);
    hold off;
end

% Trash unnecessary data.
xHist = xHist(isUsed, :);
yHist = yHist(isUsed, :);
trajStart = trajStart(isUsed, :);

trajDuration = sum(~isnan(xHist), 2);
maxTrajDuration = max(trajDuration);
xHist = xHist(:, 1:maxTrajDuration);
yHist = yHist(:, 1:maxTrajDuration);

nTraj = sum(isUsed);

toc

%%

trajEnd = trajStart + trajDuration - 1;
trajIdx = 1:nTraj;

figure;
for i = 1 : nFrames
    isVisible = (trajStart <= i & i <= trajEnd);
    col = i - trajStart + 1;
    pointX = zeros(sum(isVisible), 1);
    pointY = pointX;
    
    im = read(vidObj, frameRange(i));
    imshow(im);
    hold on;
    k = 1;
    for j = trajIdx(isVisible)
        plot(xHist(j, 1:col(j)), yHist(j, 1:col(j)), 'r-');
        pointX(k) = xHist(j, col(j));
        pointY(k) = yHist(j, col(j));
        k = k + 1;
    end
    plot(pointX, pointY, 'g.', 'MarkerSize', 1);
    hold off;
    pause(0.01);
end


%save('trajHist.mat', 'xHist', 'yHist', 'trajStart');
