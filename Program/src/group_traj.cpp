/*
group_traj.cpp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"


using namespace std;

const cv::Matx33f MAT_T(
	340.7249,	-10.9076,	-10.0000,
	52.7578,	-25.4162,	75.0000,
	0.2047,		-0.9915,	1.0000);

// inv(T) / inv(T)[3, 3]
const cv::Matx33f MAT_T_INV(
	-0.0061,	-0.0026,	0.1326,
	0.0046,		-0.0424,	3.2262,
	0.0058,		-0.0415,	1.0000);


void rectifyPoints(vector< vector<float> >& trajsX, vector< vector<float> >& trajsY, vector<bool>& isValid, int& nValid)
{
	int nTrajs = trajsX.size();
	nValid = 0;
	isValid.resize(nTrajs);


	for (int i = 0; i < nTrajs; ++i)
	{
		int trajLength = trajsX[i].size();
		bool isAllPointsValid = true;

		for (int j = 0; j < trajLength; ++j)
		{
			// [xRect; yRect; zRect] = Tinv * [x; y; 1]
			// xRect = xRect / zRect * 400
			// yRect = yRect / zRect * 400

			float x = trajsX[i][j];
			float y = trajsY[i][j];
			cv::Vec3f xyzRect = MAT_T_INV * cv::Vec3f(x, y, 1.0f);
			x = xyzRect(0) / xyzRect(2) * 400.0f;
			y = xyzRect(1) / xyzRect(2) * 400.0f;
			trajsX[i][j] = x;
			trajsY[i][j] = y;

			isAllPointsValid = (isAllPointsValid &&
								0.0f < x && x < 400.0f &&
								0.0f < y && y < 400.0f);
		}

		isValid[i] = isAllPointsValid;
		if (isAllPointsValid) ++ nValid;
	}
}


void removeInvalidPoints(const vector<bool>& isValid, const int nValid, vector< vector<float> >& trajsX, vector< vector<float> >& trajsY)
{
	int nTrajs = trajsX.size();
	vector< vector<float> > trajsXNew;
	vector< vector<float> > trajsYNew;
	trajsXNew.resize(nValid);
	trajsYNew.resize(nValid);

	for (int i = 0, j = 0; i < nTrajs; ++i)
	{
		if (isValid[i])
		{
			trajsXNew[j] = trajsX[i];
			trajsYNew[j] = trajsY[i];
			++ j;
		}
	}

	trajsX = trajsXNew;
	trajsY = trajsYNew;
}


void rectifyImage(const cv::Mat& in, cv::Mat& out)
{
	out.create(400, 400, CV_8UC3);
	const uchar* p_in = (uchar*) in.data;
	uchar* p_out = (uchar*) out.data;
	

	for (int i = 0; i < 400; ++i)
	{
		for (int j = 0; j < 400; ++j)
		{
			// [x; y; z] = T * [i/400; j/400; 1]
			// x = x / z
			// y = y / z

			float ii = static_cast<float>(i) / 400.0f;
			float jj = static_cast<float>(j) / 400.0f;
			cv::Vec3f xyz = MAT_T * cv::Vec3f(ii, jj, 1.0f);
			int x = static_cast<int>(floor(xyz(0)/xyz(2) + 0.5));
			int y = static_cast<int>(floor(xyz(1)/xyz(2) + 0.5));
			
			if (x >= 0 && x < in.cols && y >=0 && y < in.rows)
			{
				// out(j, i, :) = in(y, x, :);

				int idxIn = y*in.cols*3 + x*3;
				int idxOut = j*400*3 + i*3;

				p_out[idxOut + 0] = p_in[idxIn + 0];
				p_out[idxOut + 1] = p_in[idxIn + 1];
				p_out[idxOut + 2] = p_in[idxIn + 2];
			}
		}
	}
}


void drawAllTrajs(vector< vector<float> >& trajsX, vector< vector<float> >& trajsY, cv::Mat& im)
{
	int nTrajs = trajsX.size();
	
	for (int i = 0; i < nTrajs; ++i)
	{
		int trajSize = trajsX[i].size();
		int nLines = trajSize - 1;

		for (int j = 0; j < nLines; ++j)
		{
			float x1, y1, x2, y2;
			x1 = trajsX[i][j];
			y1 = trajsY[i][j];
			x2 = trajsX[i][j+1];
			y2 = trajsY[i][j+1];

			cv::Point start(x1, y1);
			cv::Point end(x2, y2);
			cv::line(im, start, end, CV_RGB(255,0,0));
		}
	}
}


void computeVelocities(const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, vector< vector<float> >& trajsVx, vector< vector<float> >& trajsVy)
{
	int nTrajs = trajsX.size();
	trajsVx.resize(nTrajs);
	trajsVy.resize(nTrajs);

	for (int i = 0; i < nTrajs; ++i)
	{
		int N = trajsX[i].size() - 1;
		trajsVx[i].resize(N);
		trajsVy[i].resize(N);

		for (int j = 0; j < N; ++j)
		{
			trajsVx[i][j] = trajsX[i][j+1] - trajsX[i][j];
			trajsVy[i][j] = trajsY[i][j+1] - trajsY[i][j];
		}
	}
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
	if (!videoCapture.isOpened())
	{
		cerr << "Video failed to open." << endl;
		return -1;
	}


	// -----------------------------------------
	// Begin processing video.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight;
	vector< vector<float> > trajsX, trajsY, trajsVx, trajsVy;
	vector<int> trajsStart;
	vector<bool> isValid;
	int nValid;
	int frameStart, frameEnd, frameIdx;


	getVidDim(videoCapture, vidHeight, vidWidth, nFrames);
	loadTrajectories(trajFileName, trajsX, trajsY, trajsStart, frameStart, frameEnd);
	frameIdx = frameStart;

	// Rectify trajectories, remove invalid ones, compute velocities.
	rectifyPoints(trajsX, trajsY, isValid, nValid);
	removeInvalidPoints(isValid, nValid, trajsX, trajsY);
	computeVelocities(trajsX, trajsY, trajsVx, trajsVy);

	// Compute distance and velocity similarities.
	computeSimilarities(trajsX, trajsY, trajsVx, trajsVy, posSim, velSim);



	cv::namedWindow("Frame1", cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Frame2", cv::WINDOW_AUTOSIZE);
	
	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << frameIdx << " (" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)" << endl;
		cv::Mat im, imRect;

		videoCapture.retrieve(im);
		rectifyImage(im, imRect);
		drawAllTrajs(trajsX, trajsY, imRect);
		imshow("Frame1", imRect);

		++ frameIdx;
		cv::waitKey(9000);
		break;
		//if (cv::waitKey(50) >= 0) continue;
	}


	videoCapture.release();
	return 0;
}
