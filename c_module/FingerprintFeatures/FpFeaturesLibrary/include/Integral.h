#include "opencv2\core\core.hpp"
#pragma once
using namespace cv;
class Integral
{
public:
	Integral();
	~Integral();
	int integrate(const Mat* ii, int x0, int y0, int x1, int y1);
	Mat integral_image(const Mat* img);
};

