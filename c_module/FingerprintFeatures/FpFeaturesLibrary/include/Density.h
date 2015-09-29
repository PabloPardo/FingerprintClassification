#include "opencv2\core\core.hpp"
#include "Integral.h"
#include "types.h"
using namespace std;
using namespace cv;
#pragma once
class Density
{
	Integral* integr_api;
public:
	Density();
	~Density();
	int density_pix(int, int, const Mat, int);
	Mat density_img(const Mat, int, Config* = NULL);
};

