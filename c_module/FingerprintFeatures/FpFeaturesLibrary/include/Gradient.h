#include "opencv2\core\core.hpp"
#include "types.h"
using namespace std;
#pragma once
class Gradient
{
	//FingerPrintFeatures* padre;
public:
	Gradient();
	~Gradient();
	cv::Mat gradient_pix(int i, int j, const cv::Mat* img, int radius);
	cv::Mat gradient_img(const cv::Mat* img, int radius, Config* = NULL);
};

