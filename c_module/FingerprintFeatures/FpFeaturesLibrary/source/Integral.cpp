#include "Integral.h"
#include "opencv2\imgproc\imgproc.hpp"

Integral::Integral()
{
}


Integral::~Integral()
{
}

/*
"""Use an integral image to integrate over a given window.

Parameters
----------
ii : ndarray
Integral image.
x0, y0 : int
Top - left corner of block to be summed.
x1, y1 : int
Bottom - right corner of block to be summed.

Returns
------ -
S : int
Integral(sum) over the given window.

"""
*/
int Integral::integrate(const Mat* ii, int x0, int y0, int x1, int y1)
{
	int S = 0;
	S += ii->at<int>(x1, y1);

	if ((x0 - 1 >= 0) && (y0 - 1 >= 0))
		S += ii->at<int>(x0 - 1, y0 - 1);
	if (x0 - 1 >= 0)
		S -= ii->at<int>(x0 - 1, y1);
	if (y0 - 1 >= 0)
		S -= ii->at<int>(x1, y0 - 1);
	return S;


	/*
	S = 0

	S += ii[x1, y1]

	if (x0 - 1 >= 0) and(y0 - 1 >= 0) :
	S += ii[x0 - 1, y0 - 1]

	if x0 - 1 >= 0 :
	S -= ii[x0 - 1, y1]

	if y0 - 1 >= 0 :
	S -= ii[x1, y0 - 1]

	return S
	*/
}
/*
"""Integral image / summed area table.

The integral image contains the sum of all elements above and to the
left of it, i.e.:

..math::

S[m, n] = \sum_{ i \leq m } \sum_{ j \leq n } X[i, j]

Parameters
----------
x : ndarray
	Input image.

	Returns
	------ -
S : ndarray
	Integral image / summed area table.

	References
	----------
	..[1] F.C.Crow, "Summed-area tables for texture mapping,"
	ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207 - 212.

	"""*/
Mat Integral::integral_image(const Mat* img) 
{
	Mat sum;
	integral(*img, sum);
	Mat tmp = sum.colRange(1,sum.cols);
	return tmp.rowRange(1,tmp.rows);
}


	 
