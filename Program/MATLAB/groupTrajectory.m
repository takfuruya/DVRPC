close all;
isPlotting = false;
isRecording = false;

im = read(vidObj, frameRange(1));

trajStart = traj.trajStart;
x = traj.xHist;
y = traj.yHist;
xRect = x;
yRect = y;
[nTraj, maxTrajDuration] = size(x);

% Convert trajectories to rectified coordinates.
disp('Rectifying...');
T = [
 340.7249  -10.9076  -10.0000
  52.7578  -25.4162   75.0000
   0.2047   -0.9915    1.0000 ];
Tinv = inv(T);
Tinv = Tinv / Tinv(3, 3);

imRect = rectifyImage(im, T);
for i = 1 : nTraj
    p = Tinv * [x(i, :); y(i, :); ones(1, maxTrajDuration)];
    xRect(i, :) = p(1, :) ./ p(3, :);
    yRect(i, :) = p(2, :) ./ p(3, :);
end

xRect = xRect * 400;
yRect = yRect * 400;


% Remove invalid trajectories.
isValid = ~any(xRect < 0, 2) & ~any(yRect < 0, 2) & ~any(xRect > 400, 2) & ~any(yRect > 400, 2);
%isValid(randperm(nTraj, 3000)) = false;
xRect = xRect(isValid, :);
yRect = yRect(isValid, :);
x = x(isValid, :);
y = y(isValid, :);
trajStart = trajStart(isValid, :);
nTraj = size(x, 1);


% Compute velocity.
z = zeros(nTraj, 1);
vxRect = [z diff(xRect, 1, 2)];
vyRect = [z diff(yRect, 1, 2)];


% Compute distance and velocity similarity
disp('Computing distance and velocity similarities...');
tic

trajDuration = sum(~isnan(x), 2);
trajEnd = trajStart + trajDuration - 1;
linIdxTraj = (1:nTraj)';

calcDist2 = @(a, b) (bsxfun(@minus, a', a).^2 + bsxfun(@minus, b', b).^2);
idxTriu = find(triu(ones(nTraj), 1));
nPairs = numel(idxTriu);
linIdxPairs = 1:nPairs;
posSim = spalloc(nPairs, nFrames, 100*nFrames);
velSim = spalloc(nPairs, nFrames, 100*nFrames);
isValid = spalloc(nPairs, nFrames, 100*nFrames);
accSim = spalloc(nPairs, nFrames, 100*nFrames);
for i = 1 : nFrames
    disp([num2str(i) ' / ' num2str(nFrames)]);
    
    isExist = (trajStart <= i & i <= trajEnd);
    ix = NaN(1, nTraj);
    iy = ix;
    ivx = ix;
    ivy = ix;
    
    col = i - trajStart + 1;
    idx = sub2ind([nTraj maxTrajDuration], linIdxTraj(isExist), col(isExist));
    ix(isExist) = xRect(idx);
    iy(isExist) = yRect(idx);
    ivx(isExist) = vxRect(idx);
    ivy(isExist) = vyRect(idx);
    
    delta1 = calcDist2(ix, iy);
    delta2 = calcDist2(ivx, ivy);
    dTriu1 = delta1(idxTriu);
    dTriu2 = delta2(idxTriu);
    dTriu3 = (dTriu1 < 26^2) & (dTriu2 < 3.7^2);
    iIsValid = ~isnan(dTriu1 + dTriu2);
    
    % Efficient way of doing: posSim(:, i) = dTriu1
    validRowIdx = linIdxPairs(iIsValid);
    ii = i * ones(1, sum(iIsValid));
    iPosSim = sparse(validRowIdx, ii, dTriu1(iIsValid), nPairs, nFrames);
    iVelSim = sparse(validRowIdx, ii, dTriu2(iIsValid), nPairs, nFrames);
    iAccSim = sparse(validRowIdx, ii, dTriu3(iIsValid), nPairs, nFrames);
    posSim = posSim + iPosSim;
    velSim = velSim + iVelSim;
    accSim = accSim + iAccSim;
    isValid = isValid | sparse(validRowIdx, ii, true, nPairs, nFrames);
end

toc

% Group trajectories.
disp('Grouping...');
tic
%trajSim = accSim;
trajSim = (posSim ~= 0) & (velSim ~= 0) & (posSim < sparse(26^2)) & (velSim  < sparse(3.7^2));
nTrajSim = full(sum(trajSim, 2));
nTrajValid = full(sum(isValid, 2));

trajConnectednessVec = (nTrajSim > 3) & (nTrajSim ./ nTrajValid > 0.9);
trajConnectedness = zeros(nTraj);
trajConnectedness(idxTriu) = trajConnectednessVec;
trajConnectedness = trajConnectedness | trajConnectedness' | diag(true(nTraj, 1));
disp('...Dulmage�Mendelsohn decomposition...');
[trajGroups, adjMat] = getCornerGroup(trajConnectedness, 0.5, false(1, nTraj));
toc

nGroups = max(trajGroups);
jetColors = jet(nGroups);
jetColors = shuffleRows(jetColors);
jetColors = jetColors(trajGroups, :);



%%
% Play video grouping

if ~isPlotting
    return;
end

if isRecording
    writerObj = VideoWriter('newfile.avi');
    writerObj.FrameRate = 20;
    open(writerObj);
end

hFigure = figure;
set(hFigure, 'Position', [50 50 1000 400]);

for i = 1 : nFrames
    im = read(vidObj, frameRange(i));
    isTracking = (trajStart <= i & i <= trajEnd);
    col = i - trajStart + 1;
    
    isGroupTracking = false(nGroups, 1);
    isGroupTracking(trajGroups(isTracking)) = true;
    isDone = (trajEnd < i);
    isDoneButVisible = false(nTraj, 1);
    isDoneButVisible(isDone & isGroupTracking(trajGroups)) = true;
    
    % === Plotting ===
    
    figure(hFigure);
    
    subplot(1, 2, 1);
    imshow(im);
    hold on;
    for j = linIdxTraj(isTracking)'
        plot(x(j, col(j)), y(j, col(j)), '.', 'Color', jetColors(j, :));
    end
    hold off;
    
    subplot(1, 2, 2);
    imshow(im + 255/2);
    hold on;
    for j = linIdxTraj(isDoneButVisible)'
        plot(x(j, :), y(j, :), '-', 'Color', jetColors(j, :));
    end
    for j = linIdxTraj(isTracking)'
        plot(x(j, 1:col(j)), y(j, 1:col(j)), '-', 'Color', jetColors(j, :));
    end
    hold off;
    
    
    % === Recording/Pause ===
    
    if isRecording
        frame = getframe(hFigure);
        writeVideo(writerObj, frame);
    else
        %pause;
        pause(0.01);
    end
end

if isRecording
    close(writerObj);
end


