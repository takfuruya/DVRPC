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
#include "TrajectoryListManager.h"


using namespace std;

/*
// For testing.
void displayTracking(const cv::Mat& im, const cv::Mat& bw, const cv::Mat& isValid, const TrajectoryListManager& traj)
{
	cv::Mat im2 = im.clone();
	cv::Mat bw2;
	cv::cvtColor(bw, bw2, CV_GRAY2BGR);
	
	for (int i = 0; i < traj.nTraj; ++i)
	{
		if (traj.isUsed[i])
		{
			const float* x = traj.xHist.ptr<float>(i);
			const float* y = traj.yHist.ptr<float>(i);
			int length = traj.trajLength[i];
			for (int j = 1; j < length; ++j)
			{
				cv::Point start(x[j-1], y[j-1]);
				cv::Point end(x[j], y[j]);
				cv::line(im2, start, end, CV_RGB(255,0,0));
			}
			cv::Point end(x[length-1], y[length-1]);
			cv::circle(im2, end, 2, CV_RGB(255,0,0), -1);
		}
	}

	cv::imshow("WinA", bw2);
	cv::imshow("WinB", im2);
}
*/


int main(int argc, char* argv[])
{
	if (argc != 6)
	{
		cerr << "Incorrect number of arguments." << endl;
		return -1;
	}

	string videoFileName = argv[1];
	string bgFileName = argv[2];
	string trajFileName = argv[3];
	int startTime = atoi(argv[4]);
	int duration = atoi(argv[5]);

	cv::VideoCapture videoCapture(videoFileName);
	checkInputs(videoCapture, startTime, duration);


	// -----------------------------------------
	// Initialization.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight, fps;
	int frameStart, frameEnd, frameIdx;
	vector<cv::Mat> bgMean, bgInvCov;
	cv::Mat im, imPrev, bw, isFG, isValid, isWithinBox;
	cv::Mat points, pointsPrev, pointsNew, kltStatus;
	double t0, t1; // For measuring execution speed.
	const int N_POINTS_TO_SAMPLE = 400;
	


	// Load bg and inverse covariance.
	// bgMean		(H*W) 1 x 3 (64F)
	// bgInvCov		(H*W) 3 x 3 (64F)
	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames, fps);
	loadBackground(bgFileName, bgMean, bgInvCov);


	// Specify frame range.
	frameStart = fps * startTime;
	frameEnd = frameStart + fps * duration; // 30 seconds max, 20fps.
	frameIdx = frameStart;


	// Initialize trajectory data structure.
	// Max 10000 trajectories for max 30 seconds each.
	TrajectoryListManager traj(N_POINTS_TO_SAMPLE*12, frameEnd-frameStart, frameIdx);


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------


	// Skip to first frame of interest.
	skipFrames(videoCapture, frameStart);


	// First frame.
	// im			H x W (8UC3)
	// bw			H x W (8U)
	// pointsNew	nTracking x 1 (32FC2)
	// points 		nTracking x 1 (32FC2)
	// isValid		nTracking x 1 (8U)
	// isFG			nTracking x 1 (8U)
	// isWithinBox	nTracking x 1 (8U)
	videoCapture >> im;
	imMahalDist(im, bgMean, bgInvCov, 3.0, bw); // 2.236
	getPoints(im, N_POINTS_TO_SAMPLE, pointsNew);
	traj.modify(pointsNew, points, isValid);	
	getImageValuesAt(points, bw, isFG, isWithinBox);
	isValid = isValid & isWithinBox;
	traj.add(points, isFG, isValid);
	++ frameIdx;


	//cv::namedWindow("WinA", cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("WinB", cv::WINDOW_AUTOSIZE);
	//cv::moveWindow("WinA", 1000, 50);
	//cv::moveWindow("WinB", 1000, 80+vidHeight);

	t0 = (double) cv::getTickCount();
	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << frameIdx << " (" << 100 * (frameIdx - frameStart) / (frameEnd-frameStart) << "%): ";
		cout << traj.nUsed << "/" << traj.nTraj << " (" << 100 * traj.nUsed / traj.nTraj << "%)" << endl;

		// Exchange prev/current images.
		pointsPrev = points;
		points.release();
		imPrev = im.clone();

		videoCapture.retrieve(im);
		imMahalDist(im, bgMean, bgInvCov, 3.0, bw); // 2.236
		
		// kltStatus	nTracking x 1 (8U)
		klt(imPrev, im, pointsPrev, points, kltStatus);
		isValid = isValid & kltStatus;

		
		if (frameIdx % 15 == 0)
		{
			getPoints(im, N_POINTS_TO_SAMPLE, pointsNew);
			traj.modify(pointsNew, points, isValid);
		}
		

		getImageValuesAt(points, bw, isFG, isWithinBox);
		isValid = isValid & isWithinBox;
		traj.add(points, isFG, isValid);

		
		//displayTracking(im, bw, isValid, traj);
		//if (cv::waitKey(9000) >= 0) continue;
		++ frameIdx;
	}
	t1 = (double) cv::getTickCount();


	cout << frameEnd-frameStart << " frames took ";
	cout << (t1 - t0) / cv::getTickFrequency() << "sec" << endl;


	traj.save(trajFileName);
	videoCapture.release();
	return 0;
}
