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
#include <climits>
#include "opencv2/opencv.hpp" // 2.4.8
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


void mergeGroupedTrajectories(const vector<int>& groupNumbers,
							  const int nGroups,
							  const vector<int>& trajsStart,
							  const vector< vector<float> >& trajsX,
							  const vector< vector<float> >& trajsY,
							  vector<int>& objTrajsStart,
							  vector< vector<float> >& objTrajsX,
							  vector< vector<float> >& objTrajsY,
							  vector< vector<int> >& objTrajsN)
{
	int nTrajs = groupNumbers.size();
	objTrajsStart.resize(nGroups);
	objTrajsX.resize(nGroups);
	objTrajsY.resize(nGroups);
	objTrajsN.resize(nGroups);

	for (int i = 0; i < nGroups; ++i)
	{
		int nTrajsInGroup = 0;		 	// # of trajectories in this group.
		vector<int> trajsInGroupIdx; 	// Trajectory indices of this group.
		int frameStart = INT_MAX;	 	// First frame in this group.
		int frameEnd = INT_MIN;			// Last frame in this group (exclusive).
		int nFrames = 0;				// # of frames in this group.


		// Count # of trajectories in this group.
		for (int j = 0; j < nTrajs; ++j)
		{
			if (groupNumbers[j] != i) continue;
			++ nTrajsInGroup;
		}
		

		// Get trajectory indices of this group and
		// find starting (inclusive) & ending (exclusive) frames.
		trajsInGroupIdx.resize(nTrajsInGroup);
		for (int j = 0, k = 0; j < nTrajs; ++j)
		{
			if (groupNumbers[j] != i) continue;

			int s = trajsStart[j];
			int e = s + trajsX[j].size();

			if (s < frameStart) frameStart = s;
			if (e > frameEnd) frameEnd = e;

			trajsInGroupIdx[k++] = j;
		}
		

		if (nTrajsInGroup > 0)
		{
			nFrames = frameEnd - frameStart;
		}
		objTrajsStart[i] = frameStart;
		objTrajsX[i].resize(nFrames);
		objTrajsY[i].resize(nFrames);
		objTrajsN[i].resize(nFrames);
		for (int j = 0; j < nFrames; ++j)
		{
			int frame = j + frameStart;
			float sumX = 0;
			float sumY = 0;
			int sum = 0;

			for (int k = 0; k < nTrajsInGroup; ++k)
			{
				int idx = trajsInGroupIdx[k];
				int s = trajsStart[idx];
				int e = s + trajsX[idx].size() - 1;

				if (frame < s || frame > e) continue;
				
				int m = frame - s;
				sumX += trajsX[idx][m];
				sumY += trajsY[idx][m];
				++ sum;
			}
			objTrajsX[i][j] = sumX / sum;
			objTrajsY[i][j] = sumY / sum;
			objTrajsN[i][j] = sum;
		}
	}
}


// Weighted moving average with stretched sides.
void smoothObjTrajectories(const vector< vector<float> >& objTrajsX,
						   const vector< vector<float> >& objTrajsY,
						   const vector< vector<int> >& objTrajsN,
						   vector< vector<float> >& outObjTrajsX,
						   vector< vector<float> >& outObjTrajsY)
{
	int nTrajs = objTrajsX.size();
	const int BLOCK_SIZE = 2; // 2 on each side.
	vector< vector<float> > outX;
	vector< vector<float> > outY;

	outX.resize(nTrajs);
	outY.resize(nTrajs);

	for (int i = 0; i < nTrajs; ++i)
	{
		int duration = objTrajsX[i].size();
		outX[i].resize(duration);
		outY[i].resize(duration);

		for (int j = BLOCK_SIZE; j < duration-BLOCK_SIZE; ++j)
		{
			float sumX = 0.0f;
			float sumY = 0.0f;
			int sumN = 0;

			for (int k = j-BLOCK_SIZE; k < j+BLOCK_SIZE+1; ++k)
			{
				int n = objTrajsN[i][k];
				sumX += objTrajsX[i][k] * n;
				sumY += objTrajsY[i][k] * n;
				sumN += n;
			}

			outX[i][j] = sumX / sumN;
			outY[i][j] = sumY / sumN;
		}

		// Handle edges.
		if (duration > BLOCK_SIZE * 2)
		{
			float edgeFirstX, edgeFirstY, edgeLastX, edgeLastY;
			edgeFirstX = outX[i][BLOCK_SIZE];
			edgeFirstY = outY[i][BLOCK_SIZE];
			edgeLastX = outX[i][duration-BLOCK_SIZE-1];
			edgeLastY = outY[i][duration-BLOCK_SIZE-1];

			for (int j = 0; j < BLOCK_SIZE; ++j)
			{
				outX[i][j] = edgeFirstX;
				outY[i][j] = edgeFirstY;
			}
			for (int j = duration-BLOCK_SIZE-1; j < duration; ++j)
			{
				outX[i][j] = edgeLastX;
				outY[i][j] = edgeLastY;
			}
		}
		else
		{
			float sumX = 0.0f;
			float sumY = 0.0f;
			int sumN = 0;
			float x, y;
			for (int j = 0; j < duration; ++j)
			{
				int n = objTrajsN[i][j];
				sumX += objTrajsX[i][j] * n;
				sumY += objTrajsY[i][j] * n;
				sumN += n;
			}

			x = sumX / sumN;
			y = sumY / sumN;
			for (int j = 0; j < duration; ++j)
			{
				outX[i][j] = x;
				outY[i][j] = y;
			}
		}
	}

	outObjTrajsX = outX;
	outObjTrajsY = outY;
}


// Remove first and last ends of the trajectories
// where there are only made from 1 trajectory.
void truncateObjTrajectories(const vector<int>& objTrajsStart,
							 const vector< vector<float> >& objTrajsX,
							 const vector< vector<float> >& objTrajsY,
							 const vector< vector<int> >& objTrajsN,
							 vector<int>& outObjTrajsStart,
							 vector< vector<float> >& outObjTrajsX,
							 vector< vector<float> >& outObjTrajsY)
{
	int nTrajs = objTrajsStart.size();
	vector<int> outStart;
	vector< vector<float> > outX;
	vector< vector<float> > outY;
	outStart.resize(nTrajs);
	outX.resize(nTrajs);
	outY.resize(nTrajs);

	for (int i = 0; i < nTrajs; ++i)
	{
		int duration = objTrajsX[i].size();
		int idxStart, idxEnd;	// start inclusive, end exclusive.
		idxStart = -1;
		idxEnd = -1;

		// Find idxStart.
		for (int j = 0; j < duration; ++j)
		{
			if (objTrajsN[i][j] > 1)
			{
				idxStart = j;
				break;
			}
		}

		// Find idxEnd.
		for (int j = duration; j > 0; --j)
		{
			if (objTrajsN[i][j-1] > 1)
			{
				idxEnd = j;
				break;
			}
		}

		// If the trajectory was "merged" from only one trajectory,
		// dont't consider it since it is unreliable.
		if (idxStart < 0 || idxEnd < 0) continue;

		// Beginning truncated so new starting frame.
		outStart[i] = objTrajsStart[i] + idxStart;

		// Copy.
		int size = idxEnd - idxStart;
		outX[i].resize(size);
		outY[i].resize(size);
		for (int j = idxStart, k = 0; j < idxEnd; ++j, ++k)
		{
			outX[i][k] = objTrajsX[i][j];
			outY[i][k] = objTrajsY[i][j];
		}
	}

	outObjTrajsStart = outStart;
	outObjTrajsX = outX;
	outObjTrajsY = outY;
}


/*
void markInvalidGroups(const vector< vector<float> >& objTrajsX,
					   vector<int>& groupNumbers, int& nGroups,
					   vector<int>& isValid, int& nValid)
{
	int nTrajs = groupNumbers.size();
	int nGroupsNew = nGroups;
	isValid.resize(nGroups);
	nValid = 0;


	// Count nValid and fill isValid.
	for (int i = 0; i < nGroups; ++i)
	{
		if (objTrajsX[i].size() < 1)
		{
			-- nGroupsNew;
			isValid[i] = false;

			for (int j = 0; j < nTrajs; ++j)
			{
				if (groupNumbers[j] == i) groupNumbers[j] = -1;
			}

			continue;
		}

		isValid[i] = true;
		++ nValid;
	}

	// Re-do groupNumbers
	// Copy groups such that invalid ones are -1 and group numbers are
	// from 0 to nGroups-1.
}
*/

int countVehicles(const vector< vector<float> >& x)
{
	int nGroups = x.size();
	int n = 0;

	for (int i = 0; i < nGroups; ++i)
	{
		if (x[i].size() > 0) ++ n;
	}
	return n;
}


void createColorMap(const vector<int>& groupNumbers, const int nGroups, vector<cv::Scalar>& colors, vector<cv::Scalar>& colorMap)
{
	srand (time(NULL));

	int nTrajs = groupNumbers.size();
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


//  TODO ignore -1 group
// For testing.
void drawGroupingHelper(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<cv::Scalar>& colorMap, const int frameIdx, const int lineThickness, const bool isLabel, cv::Mat& imOut)
{
	int nTrajs = trajsStart.size();
	imOut = im.clone();

	for (int i = 0; i < nTrajs; ++i)
	{
		int trajDuration = trajsX[i].size();
		int trajStart = trajsStart[i];
		
		if (trajStart > frameIdx ||
			trajStart + trajDuration < frameIdx) continue;


		int idx = frameIdx - trajStart;

		if (isLabel)
		{
			float x = trajsX[i][idx];
			float y = trajsY[i][idx];
			const int radius = 15;
			const cv::Scalar white = CV_RGB(255, 255, 255);
			const double fontScale = 0.5;
			cv::Point center(x, y);
			cv::Point textPos(x - 10.0f, y + 5.0f);
			cv::circle(imOut, center, radius, colorMap[i], lineThickness);
			cv::putText(imOut, to_string(i), textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, white, 2);
		}
		else
		{
			int nLines = idx - 1;

			for (int j = 0; j < nLines; ++j)
			{
				float x1, y1, x2, y2;
				x1 = trajsX[i][j];
				y1 = trajsY[i][j];
				x2 = trajsX[i][j+1];
				y2 = trajsY[i][j+1];

				cv::Point start(x1, y1);
				cv::Point end(x2, y2);
				cv::line(imOut, start, end, colorMap[i], lineThickness);
			}

			cv::Point center(trajsX[i][idx], trajsY[i][idx]);
			const int radius = 3;
			cv::circle(imOut, center, radius, colorMap[i], -1);
		}
	}
}


void drawGrouping(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<cv::Scalar>& colorMap, const int frameIdx, cv::Mat& imOut)
{
	drawGroupingHelper(im, trajsStart, trajsX, trajsY, colorMap, frameIdx, 1, false, imOut);
}


void drawObjects(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<cv::Scalar>& colorMap, const int frameIdx, cv::Mat& imOut)
{
	drawGroupingHelper(im, trajsStart, trajsX, trajsY, colorMap, frameIdx, 2, false, imOut);
}


void drawNumberLabel(const cv::Mat& im, const vector<int>& trajsStart, const vector< vector<float> >& trajsX, const vector< vector<float> >& trajsY, const vector<cv::Scalar>& colorMap, const int frameIdx, cv::Mat& imOut)
{
	drawGroupingHelper(im, trajsStart, trajsX, trajsY, colorMap, frameIdx, -1, true, imOut);
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
	checkInputs(videoCapture);


	// -----------------------------------------
	// Initialization.
	// -----------------------------------------


	int nFrames, vidWidth, vidHeight;
	vector< vector<float> > trajsX, trajsY;
	vector< vector<float> > trajsRectX, trajsRectY, trajsVx, trajsVy;
	vector<int> trajsStart;
	int frameStart, frameEnd, frameIdx; // Inclusive
	cv::SparseMat similarities;
	vector<int> groupNumbers;	// size = # of trajectories
	int nGroups, nVehicles;
	vector< vector<float> > objTrajsX; // size = nGroups
	vector< vector<float> > objTrajsY;
	vector<int> objTrajsStart;
	vector<cv::Scalar> colors, colorMap;
	double t0, t1, t2, t3, t4; // For measuring execution speed.


	getVidInfo(videoCapture, vidHeight, vidWidth, nFrames);
	loadTrajectories(trajFileName, trajsX, trajsY, trajsStart, frameStart, frameEnd);
	frameIdx = frameStart;


	// -----------------------------------------
	// Begin processing trajectories.
	// -----------------------------------------


	// Rectify trajectories and remove invalid ones.
	t0 = (double) cv::getTickCount();
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
	}


	// Compute velocities.
	t1 = (double) cv::getTickCount();
	computeVelocities(trajsRectX, trajsRectY, trajsVx, trajsVy);


	// Compute trajectory similarities.
	t2 = (double) cv::getTickCount();
	computeSimilarities(trajsStart, trajsRectX, trajsRectY, trajsVx, trajsVy, similarities);


	// Grouping.
	t3 = (double) cv::getTickCount();
	groupTrajectories(similarities, groupNumbers, nGroups);
	{
		vector< vector<int> > objTrajsN;
		//vector<bool> isValid;
		//int nValid;
		mergeGroupedTrajectories(groupNumbers, nGroups, trajsStart, trajsX, trajsY, objTrajsStart, objTrajsX, objTrajsY, objTrajsN);
		smoothObjTrajectories(objTrajsX, objTrajsY, objTrajsN, objTrajsX, objTrajsY);
		truncateObjTrajectories(objTrajsStart, objTrajsX, objTrajsY, objTrajsN, objTrajsStart, objTrajsX, objTrajsY);
		//markInvalidGroups(objTrajsX, groupNumbers, nGroups, isValid, nValid);
		//removeInvalidPoints(isValid, nValid, objTrajsStart);
		//removeInvalidPoints(isValid, nValid, objTrajsX);
		//removeInvalidPoints(isValid, nValid, objTrajsY);
		nVehicles = countVehicles(objTrajsX);
	}
	t4 = (double) cv::getTickCount();
	createColorMap(groupNumbers, nGroups, colors, colorMap);


	// Display results & execution times.
	cout << nVehicles << " vehicles detected." << endl;
	cout << "Execution times (sec):" << endl;
	cout << "Rectification\t" 	<< getSec(t0, t1) << endl;
	cout << "Velocity\t" 		<< getSec(t1, t2) << endl;
	cout << "Similarity\t"		<< getSec(t2, t3) << endl;
	cout << "Grouping\t"		<< getSec(t3, t4) << endl;
	cout << "Total\t\t"			<< getSec(t0, t4) << endl;


	const string FRAME_0 = "FRAME_0";
	const string FRAME_1 = "FRAME_1";
	const string FRAME_2 = "FRAME_2";
	cv::namedWindow(FRAME_0, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_0, 1000, 100);
	cv::namedWindow(FRAME_1, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_1, 1000, 600);
	cv::namedWindow(FRAME_2, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(FRAME_2, 1400, 600);


	skipFrames(videoCapture, frameStart);


	while (videoCapture.grab() && frameIdx <= frameEnd)
	{
		cout << "\r";
		cout << frameIdx << " (" << 100 * (frameIdx - frameStart) / (frameEnd - frameStart) << "%)";
		cout << "     " << flush;

		cv::Mat im, imRect, imDrawn, imRectDrawn, imRectDrawn2;

		videoCapture.retrieve(im);
		rectifyImage(im, imRect);
		
		//drawAllTrajs(trajsX, trajsY, imRect);
		//imshow("Frame1", imRect);


		//drawGrouping(im, trajsStart, trajsX, trajsY, colorMap, frameIdx, imDrawn);
		drawObjects(    	 im, objTrajsStart, objTrajsX, objTrajsY, colors, frameIdx, imDrawn);
		drawNumberLabel(imDrawn, objTrajsStart, objTrajsX, objTrajsY, colors, frameIdx, imDrawn);
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
