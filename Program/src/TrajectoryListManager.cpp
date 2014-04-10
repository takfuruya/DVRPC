#include "TrajectoryListManager.h"
#include "opencv2/opencv.hpp" // OpenCV 2.4.5
#include <iostream>
#include <fstream>
#include <climits>
#include "helper.h"

using namespace std;

TrajectoryListManager::TrajectoryListManager(int nTraj, int maxTrajDuration, int startFrame)
	: iFrame(startFrame),
	  nTraj(nTraj),
	  maxTrajDuration(maxTrajDuration)
{
	this->trajStart.reserve(nTraj);
	this->trajLength.reserve(nTraj);
	this->isUsed.reserve(nTraj);

	cv::Size size(maxTrajDuration, nTraj);
	cv::Scalar n(-1.0f);

	this->xHist = cv::Mat(size, CV_32F, n);
	this->yHist = cv::Mat(size, CV_32F, n);
	this->vxHist = cv::Mat(size, CV_32F, n);
	this->vyHist = cv::Mat(size, CV_32F, n);
	this->bHist = cv::Mat(size, CV_8U, cv::Scalar(0));

	nUsed = false;

	for (int i = 0; i < nTraj; ++i)
	{
		this->trajStart.push_back(0);
		this->trajLength.push_back(0);
		this->isUsed.push_back(false);
	}
}


void TrajectoryListManager::modify(const cv::Mat& pointsNew, cv::Mat& points, cv::Mat& isValid)
{
	int nNew = pointsNew.rows;
	int nOld = points.rows;
	int nValid = 0;

	// Accumulated trajectories (old valid trajectories + new trajectories)
	int nAcc;
	cv::Mat pointsAcc;
	vector<int> trackingIdxAcc;

	uchar* p_isValid = (uchar*) isValid.data;


	// Count total # of trajectories.
	for (int i = 0; i < nOld; ++i)
	{
		if (p_isValid[i]) ++ nValid;
	}
	nAcc = nValid + nNew;


	int j = 0;


	// Copy valid rows of points & trackingIdx to
	// pointsAcc and trackingIdxAcc. (j = 0 ... nValid-1)
	pointsAcc.create(nAcc, 1, CV_32FC2);
	trackingIdxAcc.reserve(nAcc);
	for (int i = 0; i < nOld; ++i)
	{
		if (!p_isValid[i]) continue;
		
		float* p_pointOld = points.ptr<float>(i);
		float* p_point = pointsAcc.ptr<float>(j++);
		p_point[0] = p_pointOld[0];
		p_point[1] = p_pointOld[1];
		trackingIdxAcc.push_back(trackingIdx[i]);
	}


	// At this point, j == nValid-1
	// Copy pointsNew to pointsAcc. (j = nValid-1 ... nAcc)
	int k = 0;
	for (int i = 0; i < nNew; ++i)
	{
		const float* p_pointNew = pointsNew.ptr<float>(i);
		float* p_point = pointsAcc.ptr<float>(j++);
		p_point[0] = p_pointNew[0];
		p_point[1] = p_pointNew[1];
		while (this->isUsed[k]) ++ k;
		trackingIdxAcc.push_back(k);
		this->trajStart[k] = this->iFrame;
		this->isUsed[k] = true;
		this->trajLength[k] = 0;
		++ nUsed;
	}


	this->trackingIdx = trackingIdxAcc;
	points = pointsAcc;
	isValid = cv::Mat::ones(nAcc, 1, CV_8U);
}



void TrajectoryListManager::add(const cv::Mat& points, const cv::Mat& isFG, cv::Mat& isValid)
{
	int nTracking = this->trackingIdx.size();
	const uchar* p_isFG = (uchar*) isFG.data;
	uchar* p_isValid = (uchar*) isValid.data;
	
	for (int i = 0; i < nTracking; ++i)
	{
		int idx = this->trackingIdx[i];
		int& length = trajLength[idx];

		// Mark invalid short trajectories as unneeded.
		if (!p_isValid[i])
		{
			if (length < 4 && isUsed[idx])
			{
				isUsed[idx] = false;
				-- nUsed;
			}
			continue;
		}
		
		const float* p = points.ptr<float>(i);
		float velX, velY;
		float* x = this->xHist.ptr<float>(idx);
		float* y = this->yHist.ptr<float>(idx);
		float* vx = this->vxHist.ptr<float>(idx);
		float* vy = this->vyHist.ptr<float>(idx);
		uchar* b = this->bHist.ptr<uchar>(idx);
		
		int idx1 = length - 1;
		int idx2 = length - 2;
		int idx3 = length - 3;
		float k = 1.5*1.5;

		

		//	L = 4
		//		  L-4   L-3   L-2   L-1   current
		//		  idx3  idx2  idx1
		//		|  0  |  1  |  2  |  3  |   -    |...
		//	b	|  0  |  0  |  0  |  0  | (isFG) |...
		//	x   |  x0 |  x1 |  x2 |  x3 |(points)|...
		//	vx  |  -  | x01 | x12 | x23 | (velX) |...
		

		
		// Compute velocity.
		if (length > 0)
		{
			velX = p[0] - x[idx1];
			velY = p[1] - y[idx1];
		}

		
		// Stop tracking points at the background.
		// (1) Trajectory is at least 4 frames (excluding current one).
		// (2) Last 3 frames (and current one) must be background.
		// (3) Last 3 frames (and current one) must not be moving.
		if ( (length > 3) && 					// (1)
			 !p_isFG[i] && 						// (2)
			 !b[idx1] &&
			 !b[idx2] &&
			 !b[idx3] &&
			 (velX * velX + velY * velY < k) &&	// (3)
			 (vx[idx1] * vx[idx1] + vy[idx1] * vy[idx1] < k) &&
			 (vx[idx2] * vx[idx2] + vy[idx2] * vy[idx2] < k) &&
			 (vx[idx3] * vx[idx3] + vy[idx3] * vy[idx3] < k)
		   )
		{
			// Set invalid (stop tracking) and crop last bit.
			p_isValid[i] = false;
			length -= 3;
			continue;
		}

		// Store history.
		x[length] = p[0];
		y[length] = p[1];
		b[length] = p_isFG[i];
		if (length > 0)
		{
			vx[length] = velX;
			vy[length] = velY;
		}
		++ length;
	}
	++ iFrame;
}


void TrajectoryListManager::save(const string& fileName)
{
	string sep = " ";
	ofstream file(fileName.c_str());
	std::vector<bool> isUsed2 = this->isUsed; // Make a copy to prevent modifying the original.
	int nUsed2 = 0;
	int frameStart, frameEnd, maxTrajDuration2;

	// Get rid of short trajectories & count remaining trajectories.
	for (int i = 0; i < this->nTraj; ++i)
	{
		if (!isUsed2[i]) continue;
		
		if (this->trajLength[i] < 4)
		{
			isUsed2[i] = false;
			continue;
		}

		// Count # of non-short useful trajectories.
		++ nUsed2;
	}

	// Find start frame, end frame, and maximum trajectory duration:
	frameStart = INT_MAX;
	frameEnd = INT_MIN;
	maxTrajDuration2 = INT_MIN;
	for (int i = 0; i < this->nTraj; ++i)
	{
		if (!isUsed2[i]) continue;

		int start = this->trajStart[i];
		int end = start + this->trajLength[i];
		int max = this->trajLength[i];

		if (start < frameStart) frameStart = start;
		if (end > frameEnd) frameEnd = end;
		if (max > maxTrajDuration2) maxTrajDuration2 = max;
	}


	if (!file.is_open())
	{
		cerr << "Failed to create file to save trajectories." << endl;
		exit(EXIT_FAILURE);
	}

	// Write to file.
	// Each line corresponds to a trajectory:
	// 		S L x0 y0 x1 y1 ...
	// ... where S = frame # of first point, L = # of points.
	file << nUsed2 << sep;
	file << frameStart << sep;
	file << frameEnd << sep;
	file << maxTrajDuration2 << endl;
	for (int i = 0; i < this->nTraj; ++i)
	{
		if (!isUsed2[i]) continue;

		int L = this->trajLength[i];
		float* rowX = this->xHist.ptr<float>(i);
		float* rowY = this->yHist.ptr<float>(i);

		file << this->trajStart[i] << sep;
		file << L << sep;
		for (int j = 0; j < L; ++j)
		{
			file << rowX[j] << sep;
			file << rowY[j] << sep;
		}
		file << endl;
	}

	file.close();
}