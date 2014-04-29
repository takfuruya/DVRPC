/*
extract_traj.cpp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"


using namespace std;


int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cerr << "Incorrect number of arguments." << endl;
		return -1;
	}

	string videoFileName = argv[1];
	string bgFileName = argv[2];

	cv::VideoCapture videoCapture(videoFileName);
	checkInputs(videoCapture);


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight, nPixels;
	vector<cv::Mat> bgMean, bgInvCov;


	// Load bg and inverse covariance.
	// bgMean		(H*W) 1 x 3 (64F)
	// bgInvCov		(H*W) 3 x 3 (64F)
	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames);
	cout << "Loading background." << endl;
	loadBackground(bgFileName, bgMean, bgInvCov);
	nPixels = bgMean.size();


	cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
	int iFrame = 0;
	while (videoCapture.grab())
	{
		cv::Mat im;
		cv::Mat bw(vidHeight, vidWidth, CV_8U);

		videoCapture.retrieve(im);
		imMahalDist(im, bgMean, bgInvCov, 2.236, bw);
		
		imshow("Frame", bw);
		++ iFrame;

		if (cv::waitKey(50) >= 0) break;
	}


	videoCapture.release();
	return 0;
}
