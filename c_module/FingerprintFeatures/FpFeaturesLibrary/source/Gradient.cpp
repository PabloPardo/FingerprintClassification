#include "Gradient.h"
#include "FingerPrintFeatures.h"
#include "opencv2\core\core.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Gradient::Gradient()
{
}

Gradient::~Gradient()
{
	
} 

Mat Gradient::gradient_pix(int i, int j, const Mat img, int radius)
{
	Mat ret = Mat(1,2,CV_32F);
	uchar isup = img.at<uchar>(i + radius, j);
	uchar iinf = img.at<uchar>(i - radius, j);
	uchar jsup = img.at<uchar>(i, j + radius);
	uchar jinf = img.at<uchar>(i, j - radius);
	ret.at<float>(0,0) = ((isup - iinf) / float(2 * radius));
	ret.at<float>(0,1) = ((jsup - jinf) / float(2 * radius));
	return ret;
}


Mat Gradient::gradient_img(const Mat img, int radius, Config* cfg)
{
	int m = img.rows;
	int n = img.cols;
	
	Mat grad_img = Mat(m-(2*radius), n-(2*radius), CV_32F);
	
	for (int i = radius; i < m - radius; i++) {
		for (int j = radius; j < n - radius; j++) {
			Mat grad_pix = gradient_pix(i, j, img, radius);
			double normalization = norm(grad_pix);
			grad_img.at<float>(i-radius, j-radius) = (float)normalization;
		}
	}

	return grad_img;
}

