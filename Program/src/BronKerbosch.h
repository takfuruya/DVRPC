#ifndef BRON_KERBOSCH_H
#define BRON_KERBOSCH_H

#include <vector>
#include "opencv2/opencv.hpp" // OpenCV 2.4.5

void findMaximalCliques(const cv::SparseMat& adjMat, std::vector< std::vector<int> >& groups);
void findMaximalCliques2();

#endif /* BRON_KERBOSCH_H */
