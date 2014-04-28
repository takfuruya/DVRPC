/*
group_traj.cpp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include "helper.h"
#include "BronKerbosch.h"


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

const float POS_TOLERANCE = 26.0f;
const float VEL_TOLERANCE = 3.6f;
const int MIN_SIMILAR_FRAMES = 7;


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


template<typename T>
void removeInvalidPoints(const vector<bool>& isValid, const int nValid, vector<T>& vec)
{
	int nTrajs = isValid.size();
	vector<T> vecNew;
	vecNew.resize(nValid);

	for (int i = 0, j = 0; i < nTrajs; ++i)
	{
		if (isValid[i]) vecNew[j++] = vec[i];
	}

	vec = vecNew;
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
			
			int idxOut = j*400*3 + i*3;
			if (x >= 0 && x < in.cols && y >=0 && y < in.rows)
			{
				// out(j, i, :) = in(y, x, :);

				int idxIn = y*in.cols*3 + x*3;

				p_out[idxOut + 0] = p_in[idxIn + 0]; // B.
				p_out[idxOut + 1] = p_in[idxIn + 1]; // G.
				p_out[idxOut + 2] = p_in[idxIn + 2]; // R.
			}
			else
			{
				p_out[idxOut + 0] = 0; // B.
				p_out[idxOut + 1] = 0; // G.
				p_out[idxOut + 2] = 0; // R.
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


void computeSimilarities(const vector<int>& trajsStart, const vector< vector<float> >&trajsX, const vector< vector<float> >& trajsY, const vector< vector<float> >& trajsVx, const vector< vector<float> >& trajsVy, cv::SparseMat& similarities)
{
	const int nTrajs = trajsX.size();
	const int nPairs = nTrajs * (nTrajs - 1) / 2;

	const int simDim = 2;
	int simSize[] = {nTrajs, nTrajs};
	similarities = cv::SparseMat(simDim, simSize, CV_32F);
	
	vector<int> pairsIdxA;
	vector<int> pairsIdxB;

	pairsIdxA.reserve(nPairs);
	pairsIdxB.reserve(nPairs);
	

	// Generate indices of all pairs.
	// It is an upper triangular matrix with zero diagonal assuming A is rows,
	// B is columns.
	for (int i = 0; i < nTrajs; ++i)
	{
		for (int j = i+1; j < nTrajs; ++j)
		{
			pairsIdxA.push_back(i);
			pairsIdxB.push_back(j);
		}
	}
	

	for (int i = 0; i < nPairs; ++i)
	{
		int trajIdxA = pairsIdxA[i];
		int trajIdxB = pairsIdxB[i];
		int frameFirstA = trajsStart[trajIdxA];
		int frameFirstB = trajsStart[trajIdxB];
		int frameLastA = frameFirstA + trajsX[trajIdxA].size() - 1;
		int frameLastB = frameFirstB + trajsX[trajIdxB].size() - 1;
		int frameFirst = max(frameFirstA, frameFirstB);
		int frameLast = min(frameLastA, frameLastB);

		int nSimilar = 0;
		int nOverlap = frameLast - frameFirst + 1;
		
		for (int j = frameFirst; j <= frameLast; ++j)
		{
			int idxA = j - frameFirstA;
			int idxB = j - frameFirstB;
			
			float deltaX = trajsX[trajIdxA][idxA] - trajsX[trajIdxB][idxB];
			float deltaY = trajsY[trajIdxA][idxA] - trajsY[trajIdxB][idxB];
			float deltaVx = trajsVx[trajIdxA][idxA] - trajsVx[trajIdxB][idxB];
			float deltaVy = trajsVy[trajIdxA][idxA] - trajsVy[trajIdxB][idxB];
			
			float deltaPos = deltaX * deltaX + deltaY * deltaY;
			float deltaVel = deltaVx * deltaVx + deltaVy * deltaVy;
			
			if (deltaPos < POS_TOLERANCE*POS_TOLERANCE && deltaVel < VEL_TOLERANCE*VEL_TOLERANCE)
			{
				++ nSimilar;
			}
		}
		

		if (nSimilar >= MIN_SIMILAR_FRAMES && (float)nSimilar / (float)nOverlap > 0.9f)
		{
			similarities.ref<float>(trajIdxA, trajIdxB) = 1.0f;
		}
	}
}


// Dulmage-Mendelsohn decomposition.
void groupTrajectories(const cv::SparseMat& similarities, vector<int>& groupNumbers, int& nGroups)
{
	int nTrajs = similarities.size()[0];
	vector<int> groupNumbersTemp;
	vector<bool> isUsed(nTrajs, true);
	nGroups = nTrajs;

	// Initialize all trajectories to unique group number (all separate).
	groupNumbersTemp.resize(nTrajs);
	for (int i = 0; i < nTrajs; ++i)
	{
		groupNumbersTemp[i] = i;
	}

	// Iterate through every edge to find groups.
	cv::SparseMatConstIterator_<float> it = similarities.begin<float>();
	cv::SparseMatConstIterator_<float> itEnd = similarities.end<float>();

	for (; it != itEnd; ++it)
	{
		const cv::SparseMat::Node* node = it.node();
		int trajA = node->idx[0]; // Row.
		int trajB = node->idx[1]; // Column.

		int groupA = groupNumbersTemp[trajA];
		int groupB = groupNumbersTemp[trajB];
		if (groupA == groupB) continue;
		
		// Merge A & B by converting Bs to A.
		isUsed[groupB] = false;
		-- nGroups;
		for (int i = 0; i < nTrajs; ++i)
		{
			if (groupNumbersTemp[i] == groupB)
			{
				groupNumbersTemp[i] = groupA;
			}
		}
	}

	// Remove groups with less than 4 trajectories.
	for (int i = 0; i < nTrajs; ++i)
	{
		if (!isUsed[i]) continue;

		int elmCount = 0;

		for (int j = 0; j < nTrajs; ++j)
		{
			if (i == groupNumbersTemp[j]) ++ elmCount;
		}

		if (elmCount < 4)
		{
			isUsed[i] = false;
			-- nGroups;
		}
	}

	// Copy groups such that invalid ones are -1 and group numbers are
	// from 0 to nGroups-1.

	// Initialize all to be invalid (-1).
	groupNumbers.resize(nTrajs);
	for (int i = 0; i < nTrajs; ++i)
	{
		groupNumbers[i] = -1;
	}

	for (int i = 0, j = 0; i < nTrajs; ++i)
	{
		if (!isUsed[i]) continue;

		for (int k = 0; k < nTrajs; ++k)
		{
			if (groupNumbersTemp[k] == i)
			{
				groupNumbers[k] = j;
			}
		}
		++ j;
	}
}


// Bron-Kerbosch.
void groupTrajectories2(const cv::SparseMat& similarities, vector<int>& groupNumbers, int& nGroups)
{
	int nTrajs = similarities.size()[0];
	vector< vector<int> > groups;
	cv::SparseMat similarities2;

	// Convert float to uchar
	{
		const int dim = 2;
		int size[] = {nTrajs, nTrajs};
		similarities2 = cv::SparseMat(dim, size, CV_8U);

		cv::SparseMatConstIterator_<float> it = similarities.begin<float>();
		cv::SparseMatConstIterator_<float> itEnd = similarities.end<float>();

		for (; it != itEnd; ++it)
		{
			const cv::SparseMat::Node* node = it.node();
			int i = node->idx[0]; // Row.
			int j = node->idx[1]; // Column.

			if (it.value<float>() > 0.5f)
			{
				similarities2.ref<uchar>(i, j) = 1;
			}
		}
	}

	findMaximalCliques(similarities2, groups);

	nGroups = groups.size();
	groupNumbers.resize(nTrajs);

	for (int i = 0; i < nGroups; ++i)
	{
		int nVertices = groups[i].size();

		for (int j = 0; j < nVertices; ++j)
		{
			groupNumbers[groups[i][j]] = i;
		}
	}
}


void createColorMap(const vector<int>& groupNumbers, const int nGroups, vector<cv::Scalar>& colorMap)
{
	srand (time(NULL));

	int nTrajs = groupNumbers.size();
	vector<cv::Scalar> colors;
	colors.resize(nGroups);
	colorMap.resize(nTrajs);

	
	for (int i = 0; i < nGroups; ++i)
	{
		int r = rand() % 256;
		int g = rand() % 256;
		int b = rand() % 256;
		colors[i] = CV_RGB(r, g, b);
	}

	for (int i = 0; i < nTrajs; ++i)
	{
		colorMap[i] = colors[groupNumbers[i]];
	}
}


// For testing.
void drawGrouping(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<cv::Scalar>& colorMap, const int frameIdx, cv::Mat& imOut)
{
	int nTrajs = trajsStart.size();
	imOut = im.clone();

	for (int i = 0; i < nTrajs; ++i)
	{
		int trajDuration = trajsX[i].size();
		int trajStart = trajsStart[i];
		
		if (trajStart > frameIdx ||
			trajStart + trajDuration < frameIdx) continue;

		int nLines = min(trajDuration, frameIdx-trajStart) - 1;

		for (int j = 0; j < nLines; ++j)
		{
			float x1, y1, x2, y2;
			x1 = trajsX[i][j];
			y1 = trajsY[i][j];
			x2 = trajsX[i][j+1];
			y2 = trajsY[i][j+1];

			cv::Point start(x1, y1);
			cv::Point end(x2, y2);
			cv::line(imOut, start, end, colorMap[i]);
		}
	}
}


void drawConnection(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const cv::SparseMat& similarities, const int frameIdx, cv::Mat& imOut)
{
	imOut = im.clone();

	// Iterate through every edge.
	cv::SparseMatConstIterator_<float> it = similarities.begin<float>();
	cv::SparseMatConstIterator_<float> itEnd = similarities.end<float>();

	for (; it != itEnd; ++it)
	{
		const cv::SparseMat::Node* node = it.node();
		int trajA = node->idx[0]; // Row.
		int trajB = node->idx[1]; // Column.

		int trajStartA = trajsStart[trajA];
		int trajStartB = trajsStart[trajB];
		int trajEndA = trajStartA + trajsX[trajA].size() - 1;
		int trajEndB = trajStartB + trajsX[trajB].size() - 1;

		if (frameIdx < trajStartA || trajEndA < frameIdx ||
			frameIdx < trajStartB || trajEndB < frameIdx) continue;

		int idxA = frameIdx - trajStartA;
		int idxB = frameIdx - trajStartB;
		
		// Draw line.
		float x1, y1, x2, y2;
		x1 = trajsX[trajA][idxA];
		y1 = trajsY[trajA][idxA];
		x2 = trajsX[trajB][idxB];
		y2 = trajsY[trajB][idxB];

		cv::Point start(x1, y1);
		cv::Point end(x2, y2);

		cv::line(imOut, start, end, CV_RGB(255, 0, 0));
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
	vector< vector<float> > trajsX, trajsY;
	vector< vector<float> > trajsRectX, trajsRectY, trajsVx, trajsVy;
	vector<int> trajsStart;
	int frameStart, frameEnd, frameIdx; // Inclusive
	cv::SparseMat similarities;
	vector<int> groupNumbers;	// size = # of trajectories
	int nGroups;
	vector<cv::Scalar> colorMap;


	getVidDim(videoCapture, vidHeight, vidWidth, nFrames);
	loadTrajectories(trajFileName, trajsX, trajsY, trajsStart, frameStart, frameEnd);
	frameIdx = frameStart;

	// Rectify trajectories, remove invalid ones, compute velocities.
	{
		int nValid;
		vector<bool> isValid;
		trajsRectX = trajsX;
		trajsRectY = trajsY;
		rectifyPoints(trajsRectX, trajsRectY, isValid, nValid);
		removeInvalidPoints(isValid, nValid, trajsStart);
		removeInvalidPoints(isValid, nValid, trajsX);
		removeInvalidPoints(isValid, nValid, trajsY);
		removeInvalidPoints(isValid, nValid, trajsRectX);
		removeInvalidPoints(isValid, nValid, trajsRectY);
		computeVelocities(trajsRectX, trajsRectY, trajsVx, trajsVy);
	}

	// Compute trajectory similarities.
	computeSimilarities(trajsStart, trajsRectX, trajsRectY, trajsVx, trajsVy, similarities);

	// Grouping.
	groupTrajectories(similarities, groupNumbers, nGroups);
	createColorMap(groupNumbers, nGroups, colorMap);


	const string FRAME_0 = "FRAME_0";
	const string FRAME_1 = "FRAME_1";
	const string FRAME_2 = "FRAME_2";
	cv::namedWindow(FRAME_0, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_0, 1000, 100);
	cv::namedWindow(FRAME_1, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_1, 1000, 600);
	cv::namedWindow(FRAME_2, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_2, 1400, 600);

	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << frameIdx << " (" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)" << endl;
		cv::Mat im, imRect, imDrawn, imRectDrawn, imRectDrawn2;

		videoCapture.retrieve(im);
		rectifyImage(im, imRect);
		
		//drawAllTrajs(trajsX, trajsY, imRect);
		//imshow("Frame1", imRect);

		drawGrouping(im, trajsStart, trajsX, trajsY, colorMap, frameIdx, imDrawn);
		drawGrouping(imRect, trajsStart, trajsRectX, trajsRectY, colorMap, frameIdx, imRectDrawn);
		drawConnection(imRect, trajsStart, trajsRectX, trajsRectY, similarities, frameIdx, imRectDrawn2);
		cv::imshow(FRAME_0, imDrawn);
		cv::imshow(FRAME_1, imRectDrawn);
		cv::imshow(FRAME_2, imRectDrawn2);
		
		++ frameIdx;
		//cv::waitKey(9000);
		//break;
		if (cv::waitKey(50) >= 0) continue;
	}


	videoCapture.release();
	return 0;
}
