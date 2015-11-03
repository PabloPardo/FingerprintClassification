#include "Integral.h"
#include "opencv2\imgproc\imgproc.hpp"

Integral::Integral()
{
}

Integral::~Integral()
{
}

int Integral::integrate(const Mat ii, int x0, int y0, int x1, int y1)
{
	int S = 0;
	S += ii.at<int>(x1, y1);

	if ((x0 - 1 >= 0) && (y0 - 1 >= 0))
		S += ii.at<int>(x0 - 1, y0 - 1);
	if (x0 - 1 >= 0)
		S -= ii.at<int>(x0 - 1, y1);
	if (y0 - 1 >= 0)
		S -= ii.at<int>(x1, y0 - 1);
	return S;
}

Mat Integral::integral_image(const Mat img) 
{
	Mat sum;
	integral(img, sum);
	Mat tmp = sum.colRange(1,sum.cols);
	return tmp.rowRange(1,tmp.rows);
}


	 
