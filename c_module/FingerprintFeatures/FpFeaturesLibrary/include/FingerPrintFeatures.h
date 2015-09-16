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
	float logTable[131072];
	void initLogTable(int);
public:
	Config* cfg;
	FingerPrintFeatures(Config*);
	~FingerPrintFeatures(void);
	Mat hist_density(const Mat* image, int radius, int n_bins);
	Mat hist_grad(const Mat* image, int radius, int n_bins);
	Mat diferentiate_img(const Mat* image);
	Mat hist_entropy(const Mat* img, int radius = 5, int n_bins = 64);
	float entropy(const Mat*, const Mat*);
	Mat hist_hough(const Mat* img, int n_bins);
};

