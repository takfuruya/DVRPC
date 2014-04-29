/*
play_video.cpp

*/

#include <iostream>
#include <string>
#include <cstdlib>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"


using namespace std;


int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		cerr << "Incorrect number of arguments." << endl;
		return -1;
	}

	string videoFileName = argv[1];
	int startTime = atoi(argv[2]);	// In seconds.
	int duration = atoi(argv[3]);	// In seconds.

	cv::VideoCapture videoCapture(videoFileName);
	if (!videoCapture.isOpened())
	{
		cerr << "Video failed to open." << endl;
		return -1;
	}


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------

	const int VIEW_FPS = 1;//50;
	int nFrames, vidWidth, vidHeight, fps;
	int frameStart, frameEnd, frameIdx; // Inclusive

	
	getVidDim(videoCapture, vidHeight, vidWidth, nFrames);
	fps = static_cast<int>(videoCapture.get(CV_CAP_PROP_FPS) + 0.5);
	frameStart = startTime * fps;
	frameEnd = frameStart + duration * fps;


	const string FRAME = "FRAME";
	cv::namedWindow(FRAME, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME, 1000, 100);


	// Skip frames.
	for (frameIdx = 0; frameIdx < frameStart; ++ frameIdx)
	{
		videoCapture.grab();
	}


	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << "Frame: " << frameIdx << " ";
		cout << "(" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)";
		cout << endl;
		
		cv::Mat im;
		videoCapture.retrieve(im);
		cv::imshow(FRAME, im);
		
		++ frameIdx;
		if (cv::waitKey(VIEW_FPS) >= 0) continue;
	}


	videoCapture.release();
	return 0;
}
