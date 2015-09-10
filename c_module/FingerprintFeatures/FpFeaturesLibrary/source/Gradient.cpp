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

/*
"""
Computes the gradient in a pixel using the pixels at a certain
radius.

: param i : Pixel row coordinate.
: type i : int

: param j : Pixel column coordinate.
: type j : int

: param img : Image.
: type img : ndarray

: param radius : Radius to us in the calculation.
: type radius : int

: return : Gradient Vector
"""
*/
Mat Gradient::gradient_pix(int i, int j, const Mat* img, int radius)
{
	Mat ret = Mat(1,2,CV_32F);
	uchar isup = img->at<uchar>(i + radius, j);
	uchar iinf = img->at<uchar>(i - radius, j);
	uchar jsup = img->at<uchar>(i, j + radius);
	uchar jinf = img->at<uchar>(i, j - radius);
	ret.at<float>(0,0) = ((isup - iinf) / float(2 * radius));
	ret.at<float>(0,1) = ((jsup - jinf) / float(2 * radius));
	return ret;
}

/*
"""
Computes the image of gradient magnitudes.
 
: param img : Image.
: type img : ndarray

: param radius : Radius for the gradient calculation.
: type radius : int

: return : Gradient Image
"""
*/
Mat Gradient::gradient_img(const Mat* img, int radius, Config* cfg)
{
	int m = img->rows;
	int n = img->cols;
	
	Mat grad_img = Mat(m-(2*radius), n-(2*radius), CV_32F);
	//*grad_img = Scalar(0);
	//randu(*grad_img,(float)0.0,std::numeric_limits<float>::max());

	for (int i = radius; i < m - radius; i++) {
		for (int j = radius; j < n - radius; j++) {
			Mat grad_pix = gradient_pix(i, j, img, radius);
			double normalization = norm(grad_pix);
			grad_img.at<float>(i-radius, j-radius) = normalization;
			grad_pix.release();
		}
	}

	/*
		m, n = img.shape
		grad_img = np.empty((m, n))
		for i in range(radius, m - radius) :
			for j in range(radius, n - radius) :
				grad_img[i, j] = np.linalg.norm(gradient_pix(i, j, img, radius))
		return grad_img
	*/
	return grad_img;
}

