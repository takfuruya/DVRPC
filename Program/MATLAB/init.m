close all;
clear all;
clc;

vidObj = VideoReader('../Data/1.mp4');

%%

close all;
clearvars -except vidObj;
clc;

% Load background model (bg.imMean, bg.imCov, bg.imInvCov)
bg = load('../Data/BG_unimodal_vid1_random1000.mat');

%%
close all;
clearvars -except vidObj bg;
clc;

%frameRange = 54600;    % No car
%frameRange = 14380;    % Lots of cars
%frameRange = 130;      % One car
%frameRange = 105:(105+200-1); % From top 10 sec
%frameRange = 3950:(3950+200-1); % Horizontal 10 sec
%frameRange = 4760:(4760+200-1); % Vertical 10 sec
%frameRange = 5020:(5020+200-1); % From bottom 10 sec
%frameRange = 12130:(12130+200-1); % From left 10 sec
frameRange = 15560:(15560+200-1); % From right 10 sec

nFrames = numel(frameRange);

vidHeight = vidObj.Height;
vidWidth = vidObj.Width;

rgb2labStruct = makecform('srgb2lab');
rgb2lab = @(im) applycform(im, rgb2labStruct);

hlabel = vision.ConnectedComponentLabeler;

%disp('Reading traj...');
%traj = load('trajHist_4760_4959.mat');
