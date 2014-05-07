im = imread('bg.jpg');

% [x1 y1 x2 y2]
vLines = [... % Left to right
    11  54  26  474;
    39  69  285 456;
    55  62  545 433;
    64  51  619 305];
hLines = [... % Top to bottom
    4   131 292 152;
    604 295 -1  296;
    8   389 638 350;
    358 480 639 414];

% hLines = [...
%     15  59  79  59;
%     5   117 227 122;
%     634 222 524 206;
%     632 234 537 220];

close all;
imshow(im);
hold on;
plot(vLines(:, [1 3])', vLines(:, [2 4])', 'g', 'LineWidth', 2);
plot(hLines(:, [1 3])', hLines(:, [2 4])', 'r', 'LineWidth', 2);
hold off;


%%
% Determine vanishing points from the lines
% vy = [11; 26; 1];
% vx = [1665; 258; 1];


A1 = zeros(4, 3);
A2 = zeros(4, 3);
for i = 1 : 4
    line = vLines(i, :);
    c = [line([1 2]); line([3 4])] \ [-1; -1];
    A1(i, :) = [c' 1];
end
for i = 1 : 4
    line = hLines(i, :);
    c = [line([1 2]); line([3 4])] \ [-1; -1];
    A2(i, :) = [c' 1];
end

% Find intersection
vy = minimizeAx(A1);
vy = vy / vy(3)     % Normalize
vx = minimizeAx(A2);
vx = vx / vx(3)     % Normalize


%%
% Determine transformation matrix
% T = [
%  340.7249  -10.9076  -10.0000
%   52.7578  -25.4162   75.0000
%    0.2047   -0.9915    1.0000 ]
%


%x00 = [1; 1; 1];
%x11 = [640; 480; 1];
x00 = [-10; 75; 1];
%x11 = [590; 302; 1];
x11 = [1500; 480; 1];
hold on;
plot(x00(1), x00(2), 'r.');
plot(x11(1), x11(2), 'r.');
hold off;

H = [vx vy x00];
T = H * diag(H \ x11); % Rectified to original
T = T / T(3, 3)        % Normalize


%%
% Rectify image using above transformation
%

im2 = zeros(400, 400, 3, 'uint8');

for i = 1 : 400
    for j = 1 : 400
        x = i / 400;
        y = j / 400;
        
        p = T * [x; y; 1];
        p = round(p / p(3));
        px = p(1);
        py = p(2);
        if px > 0 && px <= 640 && py > 0 && py <= 480
            im2(j, i, :) = im(py, px, :);
            
        end
    end
end


figure;
imshow(im2);