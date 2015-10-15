#include "opencv2\core\core.hpp"
#include "types.h"
using namespace std;
using namespace cv;
#pragma once
class Gradient
{
public:
	Gradient();
	~Gradient();
	cv::Mat gradient_pix(int, int, const Mat, int);
	cv::Mat gradient_img(const cv::Mat , int, Config* = NULL);
};

