#include "helper.h"
#include "opencv2/opencv.hpp" // OpenCV 2.4.5
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

void checkInputs(cv::VideoCapture& v)
{
	if (!v.isOpened())
	{
		cerr << "helper.cpp: Video failed to open." << endl;;
		exit(0);
	}
}

void checkInputs(cv::VideoCapture& v, int startTime, int duration)
{
	checkInputs(v);

	int nFrames, fps, startFrame, endFrame;
	bool isWithinRange;
	
	nFrames = static_cast<int>(v.get(CV_CAP_PROP_FRAME_COUNT) + 0.5);
	fps = static_cast<int>(v.get(CV_CAP_PROP_FPS) + 0.5);
	
	startFrame = startTime * fps;
	endFrame = startFrame + duration * fps;
	isWithinRange = (startFrame >= 0 &&
					 startFrame < endFrame &&
					 endFrame < nFrames);

	// Check range.
	if (!isWithinRange)
	{
		cerr << "helper.cpp: Video range is invalid." << endl;
		exit(0);
	}
}


void getVidInfo(cv::VideoCapture& v, int& h, int& w, int& f)
{
	double dh = v.get(CV_CAP_PROP_FRAME_HEIGHT);
	double dw = v.get(CV_CAP_PROP_FRAME_WIDTH);
	double df = v.get(CV_CAP_PROP_FRAME_COUNT);
	h = static_cast<int>(dh + 0.5);
	w = static_cast<int>(dw + 0.5);
	f = static_cast<int>(df + 0.5);
}


void getVidInfo(cv::VideoCapture& v, int& h, int& w, int& f, int& r)
{
	getVidInfo(v, h, w, f);
	
	double dr = v.get(CV_CAP_PROP_FPS);
	r = static_cast<int>(dr + 0.5);
}


void skipFrames(cv::VideoCapture& v, int f)
{
	for (int i = 0; i < f; ++i)
	{
		v.grab();
	}
}


void loadBackground(const string& fileName, vector<cv::Mat>& bgMean, vector<cv::Mat>& bgCov)
{
	string str;
	int nPixels;
	ifstream file(fileName.c_str());
	
	if (!file.is_open())
	{
		cerr << "Failed to open background file." << endl;
		exit(EXIT_FAILURE);
	}

	// Read nPixels and resize bgMean, bgCov.
	file >> str;
	nPixels = atoi(str.c_str());
	bgMean.reserve(nPixels);
	bgCov.reserve(nPixels);

	// Fill bgMean and bgCov.
	for (int i = 0; i < nPixels; ++i)
	{
		cv::Mat mean(1, 3, CV_64F);
		cv::Mat cov(3, 3, CV_64F);

		double* pA = (double*) mean.data;
		double* pB = (double*) cov.data;

		// Read mean.
		file >> str;
		pA[0] = atof(str.c_str()); // B
		file >> str;
		pA[1] = atof(str.c_str()); // G
		file >> str;
		pA[2] = atof(str.c_str()); // R

		// Read cov.
		file >> str;
		pB[0] = atof(str.c_str());
		file >> str;
		pB[1] = pB[3] = atof(str.c_str());
		file >> str;
		pB[2] = pB[6] = atof(str.c_str());
		file >> str;
		pB[4] = atof(str.c_str());
		file >> str;
		pB[5] = pB[7] = atof(str.c_str());
		file >> str;
		pB[8] = atof(str.c_str());

		bgMean.push_back(mean);
		bgCov.push_back(cov);
	}

	file.close();
}


void imMahalDist(const cv::Mat& im, const vector<cv::Mat>& bgMean, const vector<cv::Mat>& bgInvCov, const double& thresh, cv::Mat& bw)
{
	int h = im.rows;
	int w = im.cols;
	bw.create(h, w, CV_8U);


	int k = 0;
	for (int i=0; i < h; ++i)
	{
		const uchar* p = im.ptr<uchar>(i);
		uchar* q = bw.ptr<uchar>(i);

		for (int j=0; j < w; ++j)
		{
			cv::Matx13d m(p[j*3], p[j*3 + 1], p[j*3 + 2]);
			double val = Mahalanobis(m, bgMean[k], bgInvCov[k]);

			// Threshold.
			if (val < thresh)
			{
				q[j] = 255;
			}
			else
			{
				q[j] = 0;
			}

			++ k;
		}
	}

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(bw, bw, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, kernel);
	bw = 255 - bw;
}


void getPoints(const cv::Mat& frame, int nPoints, cv::Mat& points)
{
	cv::Mat frameGray;
	cv::cvtColor(frame, frameGray, CV_RGB2GRAY);
	cv::goodFeaturesToTrack(frameGray, points, nPoints, 0.01, 5);
}


void getImageValuesAt(const cv::Mat& points, const cv::Mat& bw, cv::Mat& isFG, cv::Mat& isWithin)
{
	int n = points.rows;
	int w = bw.cols;
	int h = bw.rows;
	isFG.create(n, 1, CV_8U);
	isWithin.create(n, 1, CV_8U);

	uchar* p0 = (uchar*) bw.data;
	uchar* p1 = (uchar*) isFG.data;
	uchar* p2 = (uchar*) isWithin.data;

	cv::Mat points16U;
	points.convertTo(points16U, CV_16U);

	for (int i = 0; i < n; ++i)
	{
		const ushort* p = points16U.ptr<ushort>(i);
		ushort x = p[0];
		ushort y = p[1];

		if (y < h && x < w)
		{
			p1[i] = p0[y*w + x];
			p2[i] = true;
		}
		else
		{
			p1[i] = 0;
			p2[i] = false;
		}
	}
}


void dispMatInfo(const cv::Mat& m)
{
	cout << m.rows << "x" << m.cols << "x" << m.channels();
	cout << " (";

	switch(m.depth())
	{
		case CV_8U:
			cout << "uchar";
			break;
		case CV_8S:
			cout << "schar";
			break;
		case CV_16U:
			cout << "ushort";
			break;
		case CV_16S:
			cout << "short";
			break;
		case CV_32S:
			cout << "int";
			break;
		case CV_32F:
			cout << "float";
			break;
		case CV_64F:
			cout << "double";
			break;
	}
	cout << ") ";
	
	if (m.isContinuous())
	{
		cout << "Continuous";
	}
	else
	{
		cout << "Not Continuous";
	}

	cout << endl;
	return;
}

void klt(const cv::Mat& prevImg, const cv::Mat& nextImg, const cv::Mat& prevPts,
		 cv::Mat& nextPts, cv::Mat& status)
{
	// err			nTracking x 1 (32F)
	// nextNextPts	nTracking x 1 (32FC2)
	// status1		nTracking x 1 (8U)
	// status2		nTracking x 1 (8U)
	cv::Mat err;
	cv::Mat nextNextPts;
	cv::Mat status1, status2;
	int nPts = prevPts.rows;
	uchar* p_status;
	float* p_pts1;
	float* p_pts2;

	// KLT parameters.
	const float maxBidirectionalError = 2.0*2.0;
	cv::Size winSize(31, 31);
	int maxLevel = 3;
	cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
	int flags = 0;
	double minEigThreshold = 1e-4;


	calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status1, err, winSize, maxLevel, criteria, flags, minEigThreshold);
	calcOpticalFlowPyrLK(nextImg, prevImg, nextPts, nextNextPts, status2, err, winSize, maxLevel, criteria, flags, minEigThreshold);
	status = status1 & status2;

	// Calculate bidirectional error (see MATLAB vision.PointTracker doc).
	p_status = (uchar*) status.data;
	p_pts1 = (float*) prevPts.data;
	p_pts2 = (float*) nextNextPts.data;
	for (int i = 0; i < nPts; ++i)
	{
		if (!p_status[i]) continue;

		int xIdx = i*2;
		int yIdx = xIdx + 1;

		float deltaX, deltaY, r;
		deltaX = p_pts1[xIdx] - p_pts2[xIdx];
		deltaY = p_pts1[yIdx] - p_pts2[yIdx];
		r = deltaX * deltaX + deltaY * deltaY;
		p_status[i] = (r < maxBidirectionalError);
	}
}

void loadTrajectories(const string& fileName, vector< vector<float> >& trajsX, vector< vector<float> >& trajsY, vector<int>& trajsStart, int& frameStart, int& frameEnd)
{
	string str;
	int nTrajs;
	ifstream file(fileName.c_str());
	
	if (!file.is_open())
	{
		cerr << "Failed to open trajectories file." << endl;
		exit(EXIT_FAILURE);
	}


	// Read nTrajs, frameStart, frameEnd.
	file >> str;
	nTrajs = atoi(str.c_str());
	file >> str;
	frameStart = atoi(str.c_str());
	file >> str;
	frameEnd = atoi(str.c_str());
	file >> str; // Not using maxTrajDuration.


	// Fill trajs each as (x0, y0, x1, y1, x2, y2, ...).
	trajsX.resize(nTrajs);
	trajsY.resize(nTrajs);
	trajsStart.resize(nTrajs);
	for (int i = 0; i < nTrajs; ++i)
	{
		int nPts, start;

		// Read start frame and # of points.
		file >> str;
		start = atoi(str.c_str());
		file >> str;
		nPts = atoi(str.c_str());

		trajsStart[i] = start;
		trajsX[i].resize(nPts);
		trajsY[i].resize(nPts);

		// Read x & y.
		for (int j = 0; j < nPts; ++j)
		{
			file >> str;
			trajsX[i][j] = atof(str.c_str()); 	// x
			file >> str;
			trajsY[i][j] = atof(str.c_str()); 	// y
		}
	}

	file.close();
}


double getSec(const double& t0, const double& t1)
{
	return ( (t1 - t0) / cv::getTickFrequency() );
}

