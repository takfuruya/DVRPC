
#include "BronKerbosch.h"
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	const int dim = 2;
	int size[] = {8, 8};
	cv::SparseMat adjMat(dim, size, CV_8U);
	vector< vector<int> > groups;

	adjMat.ref<uchar>(0, 1) = 1;
	adjMat.ref<uchar>(0, 2) = 1;
	adjMat.ref<uchar>(0, 3) = 1;
	adjMat.ref<uchar>(1, 2) = 1;
	adjMat.ref<uchar>(1, 3) = 1;
	adjMat.ref<uchar>(2, 3) = 1;
	adjMat.ref<uchar>(3, 4) = 1;
	adjMat.ref<uchar>(4, 5) = 1;
	adjMat.ref<uchar>(5, 6) = 1;
	adjMat.ref<uchar>(5, 7) = 1;
	adjMat.ref<uchar>(6, 7) = 1;

	findMaximalCliques(adjMat, groups);
	
	int nGroups = groups.size();
	
	cout << nGroups << " group(s)." << endl;

	for (int i = 0; i < nGroups; ++i)
	{
		int nVertices = groups[i].size();
		for (int j = 0; j < nVertices; ++j)
		{
			cout << groups[i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}