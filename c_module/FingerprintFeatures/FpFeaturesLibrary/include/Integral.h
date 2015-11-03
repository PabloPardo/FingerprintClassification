#include "opencv2\core\core.hpp"
#pragma once
using namespace cv;
class Integral
{
public:
	Integral();
	~Integral();
	int integrate(const Mat, int, int, int, int);
	Mat integral_image(const Mat);
};

