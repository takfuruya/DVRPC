/*
test_traj_extraction.cpp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"


using namespace std;


void displayTracking(const cv::Mat& im, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<int>& trajsStart, int frameIdx)
{
	cv::Mat im2 = im.clone();
	int nTrajs = trajsX.size();

	for (int i = 0; i < nTrajs; ++i)
	{
		int trajSize = trajsX[i].size();
		int trajStart = trajsStart[i];
		
		if (trajStart > frameIdx ||
			trajStart + trajSize < frameIdx) continue;

		int nLines = min(trajSize, frameIdx-trajStart) - 1;

		for (int j = 0; j < nLines; ++j)
		{
			float x1, y1, x2, y2;
			x1 = trajsX[i][j];
			y1 = trajsY[i][j];
			x2 = trajsX[i][j+1];
			y2 = trajsY[i][j+1];

			cv::Point start(x1, y1);
			cv::Point end(x2, y2);
			cv::line(im2, start, end, CV_RGB(255,0,0));
		}
	}

	cv::imshow("Frame", im2);
}


int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cerr << "Incorrect number of arguments." << endl;
		return -1;
	}

	string videoFileName = argv[1];
	string trajFileName = argv[2];

	cv::VideoCapture videoCapture(videoFileName);
	checkInputs(videoCapture);


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight;
	vector< vector<float> > trajsX, trajsY;
	vector<int> trajsStart;
	int frameStart, frameEnd, frameIdx;


	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames);
	loadTrajectories(trajFileName, trajsX, trajsY, trajsStart, frameStart, frameEnd);
	frameIdx = frameStart;

	cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
	
	skipFrames(videoCapture, frameStart);

	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << frameIdx << " (" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)" << endl;
		cv::Mat im;

		videoCapture.retrieve(im);
		displayTracking(im, trajsX, trajsY, trajsStart, frameIdx);

		++ frameIdx;
		if (cv::waitKey(50) >= 0) break;
	}


	videoCapture.release();
	return 0;
}
