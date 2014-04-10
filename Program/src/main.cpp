#include <iostream>
#include <string>
#include "opencv2/opencv.hpp" // 2.4.5 pkg-config --modversion opencv


using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
	std::string fileName = argv[1];
	VideoCapture videoCapture(fileName);
	if (!videoCapture.isOpened())
	{
		std::cout << "Video failed to open." << std::endl;
		return -1;
	}

	//namedWindow("Frame", WINDOW_AUTOSIZE);
	//imshow("Frame", frameDisp);

	/*
	int history = 20;
	int nmixtures = 2;
	double backgroundRatio = 0.1;
	double noiseSigma = 30*0.5;

	BackgroundSubtractorMOG subtractor(history, nmixtures, backgroundRatio, noiseSigma);
	Mat frame;
	Mat fgmask;
	namedWindow("Frame", WINDOW_AUTOSIZE);
	namedWindow("Mask", WINDOW_AUTOSIZE);


	for(;;)
	{
		videoCapture >> frame;
		
		subtractor.operator()(frame, fgmask, 0.09);
		cv::erode(fgmask, fgmask, cv::Mat());
		cv::dilate(fgmask, fgmask, cv::Mat());

		imshow("Frame", frame);
		imshow("Mask", fgmask);

		//std::cout << minMaxLoc(fgmask, )

		if(waitKey(50) >= 0) break;
	}
	*/

	for (;;)
	{
		if (waitKey(100) >= 0) break;
	}

	return 0;
}
