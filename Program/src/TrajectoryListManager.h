#ifndef TRAJECTORY_LIST_MANAGER_H
#define TRAJECTORY_LIST_MANAGER_H

#include <vector>
#include "opencv2/opencv.hpp" // OpenCV 2.4.5

class TrajectoryListManager
{
	public:
		// Constructor
		// Memory usage ~= 17 x nTraj x maxTrajDuration bytes
		TrajectoryListManager(int nTraj, int maxTrajDuration, int startFrame);
		
		// pointsNew	nTracking x 1 (32FC2)
		// points		nTracking x 1 (32FC2)
		// isValid		nTracking x 1 (8U)
		void modify(const cv::Mat& pointsNew, cv::Mat& points, cv::Mat& isValid);

		// points	nTracking x 1 (32FC2)
		// isFG		nTracking x 1 (8U)
		// isValid	nTracking x 1 (8U)
		void add(const cv::Mat& points, const cv::Mat& isFG, cv::Mat& isValid);

		void save(const std::string& fileName);

	//private:
		int iFrame; 					// Current frame number.
		int nTraj;						// # of trajectories (capacity).
		int maxTrajDuration;			// Max # of frames a trajectory can have.
		std::vector<int> trajStart;		// (size: nTraj) Frame # trajectory starts.
		std::vector<int> trajLength;	// (size: nTraj) Each traj's # of frames.
		std::vector<bool> isUsed;		// (size: nTraj)
		int nUsed;						// # of trues in isUsed.
		cv::Mat xHist;					// nTraj x maxTrajDuration (32F)
		cv::Mat yHist;					// nTraj x maxTrajDuration (32F)
		cv::Mat vxHist;					// nTraj x maxTrajDuration (32F)
		cv::Mat vyHist;					// nTraj x maxTrajDuration (32F)
		cv::Mat bHist;					// nTraj x maxTrajDuration (8U)
		std::vector<int> trackingIdx;	// (size: nTracking)
};


#endif /* TRAJECTORY_LIST_MANAGER_H */
