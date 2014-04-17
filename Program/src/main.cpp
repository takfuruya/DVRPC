#include <iostream>
#include <string>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv


using namespace std;


int main(int argc, char* argv[])
{
	string fileName = argv[1];
	cv::VideoCapture videoCapture(fileName);
	if (!videoCapture.isOpened())
	{
		cout << "Video failed to open." << endl;
		return -1;
	}

	// Comments indicate OpenCV defaults.
	int history = 20; // 200
	int nmixtures = 2; // 5
	double backgroundRatio = 0.1; // 0.7
	double noiseSigma = 30*0.5; // 15

	cv::BackgroundSubtractorMOG subtractor(history, nmixtures, backgroundRatio, noiseSigma);
	cv::Mat frame;
	cv::Mat fgmask;
	cv::namedWindow("Frame0", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Frame0", 1000, 50);
	cv::namedWindow("Frame1", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Frame1", 1000, 550);


	for(;;)
	{
		videoCapture >> frame;
		
		subtractor.operator()(frame, fgmask, 0.09);
		cv::erode(fgmask, fgmask, cv::Mat());
		cv::dilate(fgmask, fgmask, cv::Mat());

		cv::imshow("Frame0", frame);
		cv::imshow("Frame1", fgmask);

		//cout << minMaxLoc(fgmask, )

		if(cv::waitKey(50) >= 0) break;
	}
	
	/*
	for (;;)
	{
		if (waitKey(100) >= 0) break;
	}
	*/

	return 0;
}
