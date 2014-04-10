clear all;
close all;
clc;

%% Load trajectories

fid = fopen('../intermediate/traj.txt');
getFileLine = @(f) cell2mat(textscan(fgets(f), '%f32'));

% Parse file header.
val = getFileLine(fid);
nTrajs = val(1);
frameStart = val(2);
frameEnd = val(3);
maxTrajDuration = val(4);

% Initialize matrix containing trajectories.
x = NaN(nTrajs, maxTrajDuration);
y = x;
trajsStart = NaN(nTrajs, 1);

for i = 1 : nTrajs
    val = getFileLine(fid);
    trajsStart(i) = val(1);
    trajDur = val(2);
    traj = reshape(val(3:end), 2, trajDur); % [x x x ...; y y y ...]
    x(i, 1:trajDur) = traj(1, :);
    y(i, 1:trajDur) = traj(2, :);
end

fclose(fid);

%%

xRect = x;
yRect = y;

T = [
 340.7249  -10.9076  -10.0000
  52.7578  -25.4162   75.0000
   0.2047   -0.9915    1.0000 ];
Tinv = inv(T);
Tinv = Tinv / Tinv(3, 3);


for i = 1 : nTrajs
    p = Tinv * [x(i, :); y(i, :); ones(1, maxTrajDuration)];
    xRect(i, :) = p(1, :) ./ p(3, :);
    yRect(i, :) = p(2, :) ./ p(3, :);
end

xRect = xRect * 400;
yRect = yRect * 400;


%%

imRect = imread('imRect.jpg');
imshow(imRect);
hold on;
plot(xRect', yRect');
hold off;

