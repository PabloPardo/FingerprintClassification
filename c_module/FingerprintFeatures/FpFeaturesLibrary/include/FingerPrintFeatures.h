#include "opencv2\core\core.hpp"
#include "Gradient.h"
#include "Density.h"
#include "types.h"
using namespace cv;
#pragma once
class FingerPrintFeatures
{
	Gradient* grad_api;
	Density* dens_api;
public:
	Config* cfg;
	FingerPrintFeatures(Config*);
	~FingerPrintFeatures(void);
	void hist_density(Mat*, const Mat, int, int);
	void hist_grad(Mat*, const Mat, int, int);
	void diferentiate_img(Mat*, const Mat);
	void hist_entropy(Mat*, const Mat, int, int);
	float entropy(const Mat, const Mat);
	void hist_hough(Mat*, const Mat, int);
};

