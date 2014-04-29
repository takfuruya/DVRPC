/*
extract_bg.cpp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"


using namespace std;


// Get k distinct random numbers in the range [0..n-1]
// x must have allocated memory of length k. Requires n > k.
// Recommended n/2 > k.
void getDistinctRandNum(int k, int n, int* x)
{
	bool* isTaken = new bool[n]();

	for (int i = 0; i < k; ++i)
	{
		// Get a random number that is not already taken.
		int t;
		do
		{
			t = rand() % n;
		} while (isTaken[t]);
		
		isTaken[t] = true;
		x[i] = t;
	}

	delete[] isTaken;
}


// m	H x W (8UC3) continuous
// row	Within range [0, N]
// v	(H*W) N x 3 (8U) continuous
//
// --- Operation ---
// bgr = m(i, j)
// v[i*w + j](row, :) = bgr
void copyMatrixRearranged(const cv::Mat& m, int row, vector<cv::Mat>& v)
{
	int nRows = m.rows;
	int nCols = m.cols;
	int k = 0;

	for (int i = 0; i < nRows; ++i)
	{
		const uchar* p = m.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j)
		{
			uchar* q = v[k].ptr<uchar>(row);
			q[0] = p[j*3];		// B
			q[1] = p[j*3+1];	// G
			q[2] = p[j*3+2];	// R
			++ k;
		}
	}
}


// src		(H*W) N x 3 (8U)
// bgMean	(H*W) 1 x 3 (64F)
// bgCov	(H*W) 3 x 3 (64F)
void fitGaussian(const vector<cv::Mat>& src, vector<cv::Mat>& bgMean, vector<cv::Mat>& bgCov)
{
	int nPixels = src.size();
	int covFlags = CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE;

	// Compute covariance for each pixel.
	for (int i = 0; i < nPixels; ++i)
	{
		cv::Mat m;
		src[i].convertTo(m, CV_64F);
		cv::calcCovarMatrix(m, bgCov[i], bgMean[i], covFlags, CV_64F);
	}

	// For test/debug:
	/*
	int h = 480;
	int w = 640;
	cv::Mat bgMean2(h, w, CV_64FC3);

	// Reshape bgMean to bgMean2.
	int k = 0;
	for (int i = 0; i < h; ++i)
	{
		double* p2 = bgMean2.ptr<double>(i);

		for (int j = 0; j < w; ++j)
		{
			double* p = (double*) bgMean[k].data;
			p2[j*3] = p[0];		// B
			p2[j*3 + 1] = p[1];	// G
			p2[j*3 + 2] = p[2]; // R
			++ k;
		}
	}
	
	// Display.
	bgMean2.convertTo(bgMean2, CV_8UC3);
	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test", bgMean2);
	cv::waitKey(9000);
	*/
}


// TODO 1: make non-positive definite matrix positive definite
// before inverting. Non-pos def matrix is caused by numerical error.
// TODO 2: take care close-to-singular cases.
void makePosDevAndInvert(vector<cv::Mat>& bgCov)
{
	int nPixels = bgCov.size();
	for (int i = 0; i < nPixels; ++i)
	{
		invert(bgCov[i], bgCov[i], cv::DECOMP_SVD);
	}
}


void saveBackground(const string& fileName, const vector<cv::Mat>& bgMean, const vector<cv::Mat>& bgCov)
{
	string sep = " ";
	int nPixels = bgMean.size();
	ofstream file(fileName.c_str());

	if (!file.is_open())
	{
		cerr << "Failed to create file to save background." << endl;
		exit(EXIT_FAILURE);
	}

	// Write mean and upper triangular part of covariance matrix (since it is symmetric).
	// Each line corresponds to a pixel.
	file << nPixels << endl;
	for (int i = 0; i < nPixels; ++i)
	{
		double* p;

		// Mean BGR.
		p = (double*) bgMean[i].data;
		file << p[0] << sep << p[1] << sep << p[2] << sep;
		
		// Cov.
		p = (double*) bgCov[i].data;
		file << p[0] << sep << p[1] << sep << p[2] << sep;
		file << p[4] << sep << p[5] << sep << p[8] << endl;
	}

	file.close();
}


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
	int nRandom = 1000;
	vector<int> randomNumbers(nRandom, 0);
	vector<cv::Mat> randomFrames, bgCov, bgMean;
	double t0, t1, t2;


	// Initialize matrices.
	// bgMean		(H*W) 1 x 3 (64F)
	// bgCov		(H*W) 3 x 3 (64F)
	// randomFrames	(H*W) N x 3 (8U)
	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames);
	nPixels = vidHeight * vidWidth;
	bgMean.reserve(nPixels);
	bgCov.reserve(nPixels);
	randomFrames.reserve(nPixels);
	for (int i = 0; i < nPixels; ++i)
	{
		bgMean.push_back(cv::Mat(1, 3, CV_64F));
		bgCov.push_back(cv::Mat(3, 3, CV_64F));
		randomFrames.push_back(cv::Mat(nRandom, 3, CV_8U));
	}


	// Create a list of sorted random numbers.
	srand(time(0));
	getDistinctRandNum(nRandom, nFrames, &randomNumbers[0]);
	sort(randomNumbers.begin(), randomNumbers.end());


	// For test/debug:
	/*
	int* x = new int[10];
	getDistinctRandNum(10, 20, x);
	sort(x, x+10);

	for (int i = 0; i < 10; ++i)
		cout << x[i] << " ";
	cout << endl;
	*/


	cout << "Video contains " << nFrames << " frames." << endl;
	cout << "Extracting " << nRandom << " random frames:" << endl;


	// Extract random frames.
	t0 = (double) cv::getTickCount();
	int iFrame = 0;
	int iRandomNumber = 0;
	while (videoCapture.grab() && (iRandomNumber < nRandom))
	{
		if (iFrame == randomNumbers[iRandomNumber])
		{
			cv::Mat m;
			videoCapture.retrieve(m);
			copyMatrixRearranged(m, iRandomNumber, randomFrames);

			++ iRandomNumber;
			cout << "\r" << iRandomNumber * 100 / nRandom;
			cout << "%    " << flush;
		}
		++ iFrame;
	}
	videoCapture.release();
	t1 = (double) cv::getTickCount();
	cout << getSec(t0, t1) << "sec" << endl;


	// Calculate mean & inverse covariance.
	cout << "Computing inverse covariance." << endl;
	fitGaussian(randomFrames, bgMean, bgCov);
	makePosDevAndInvert(bgCov);


	t2 = (double) cv::getTickCount();
	cout << getSec(t1, t2) << "sec" << endl;


	// Save background model to file.
	// NOTE: currently it saves double with precision 6 so some info is lost,
	// but using more precision increases file size. 
	cout << "Saving background to file." << endl;
	saveBackground(bgFileName, bgMean, bgCov);


	return 0;
}
