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
	if (argc != 4)
	{
		cerr << "Incorrect number of arguments." << endl;
		return -1;
	}

	string videoFileName = argv[1];
	string bgFileName = argv[2];
	string trajFileName = argv[3];

	cv::VideoCapture videoCapture(videoFileName);
	if (!videoCapture.isOpened())
	{
		cerr << "Video failed to open." << endl;
		return -1;
	}


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight;
	vector<cv::Mat> bgMean, bgInvCov;
	cv::Mat im, imPrev, bw, isFG, isValid, isWithinBox;
	cv::Mat points, pointsPrev, pointsNew, kltStatus;
	const int N_POINTS_TO_SAMPLE = 400;
	const int START_FRAME = 0;
	const int END_FRAME = 20*30;	// 30 seconds, 20fps.
	int iFrame = START_FRAME;
	double t0, t1;


	// Load bg and inverse covariance.
	// bgMean		(H*W) 1 x 3 (64F)
	// bgInvCov		(H*W) 3 x 3 (64F)
	getVidDim(videoCapture, vidHeight, vidWidth, nFrames);
	loadBackground(bgFileName, bgMean, bgInvCov);


	// Initialize trajectory data structure.
	// Max 10000 trajectories for max 30 seconds each.
	TrajectoryListManager traj(N_POINTS_TO_SAMPLE*12, END_FRAME-START_FRAME, iFrame);


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
	++ iFrame;


	//cv::namedWindow("WinA", cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("WinB", cv::WINDOW_AUTOSIZE);
	//cv::moveWindow("WinA", 1000, 50);
	//cv::moveWindow("WinB", 1000, 80+vidHeight);

	t0 = (double) cv::getTickCount();
	while (videoCapture.grab() && iFrame <= END_FRAME)
	{
		cout << iFrame << " (" << 100 * (iFrame - START_FRAME) / (END_FRAME-START_FRAME) << "%): ";
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

		
		if (iFrame % 15 == 0)
		{
			getPoints(im, N_POINTS_TO_SAMPLE, pointsNew);
			traj.modify(pointsNew, points, isValid);
		}
		

		getImageValuesAt(points, bw, isFG, isWithinBox);
		isValid = isValid & isWithinBox;
		traj.add(points, isFG, isValid);

		
		//displayTracking(im, bw, isValid, traj);
		//if (cv::waitKey(9000) >= 0) continue;
		++ iFrame;
	}
	t1 = (double) cv::getTickCount();


	cout << END_FRAME-START_FRAME << " frames took ";
	cout << (t1 - t0) / cv::getTickFrequency() << "sec" << endl;


	traj.save(trajFileName);
	videoCapture.release();
	return 0;
}
