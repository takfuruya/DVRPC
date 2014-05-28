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
	checkInputs(videoCapture, startTime, duration);


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------

	const int VIEW_FPS = 1;//50;
	int nFrames, vidWidth, vidHeight, fps;
	int frameStart, frameEnd, frameIdx; // Inclusive

	
	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames, fps);
	frameStart = startTime * fps;
	frameEnd = frameStart + duration * fps;
	frameIdx = frameStart;


	const string FRAME = "FRAME";
	cv::namedWindow(FRAME, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME, 1000, 100);


	// Skip frames.
	skipFrames(videoCapture, frameStart);


	// Open video for writing.
	cv::VideoWriter videoWriter;
	string outVideoFileName = "/home/takashi/Documents/DVRPC/out.avi";
	videoWriter.open(outVideoFileName,
					 CV_FOURCC('P','I','M','1'),
					 static_cast<double>(fps),
					 cv::Size(vidWidth, vidHeight));
	if (!videoWriter.isOpened())
	{
		cerr << "Could not open the output video for write." << endl;
		return -1;
	}


	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << "Frame: " << frameIdx << " ";
		cout << "(" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)";
		cout << endl;
		
		cv::Mat im;
		videoCapture.retrieve(im);
		cv::imshow(FRAME, im);
		
		videoWriter << im;

		++ frameIdx;
		if (cv::waitKey(VIEW_FPS) >= 0) continue;
	}


	videoCapture.release();
	return 0;
}
