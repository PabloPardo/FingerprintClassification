#include "opencv2\core\core.hpp"
#include "Integral.h"
#include "types.h"
using namespace std;
using namespace cv;
#pragma once
class Density
{
	//FingerPrintFeatures* padre;
	Integral* integr_api;
public:
	Density();
	~Density();
	int density_pix(int i, int j, const Mat* ii, int sz_area);
	Mat density_img(const cv::Mat* img, int radius, Config* = NULL);
};

