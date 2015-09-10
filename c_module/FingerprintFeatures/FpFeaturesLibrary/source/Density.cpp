#include "Density.h"
#include "Integral.h"
#include "FingerPrintFeatures.h"

Density::Density()
{
	integr_api = new Integral();
}


Density::~Density()
{
}

/*
	Computes the density around the pixel (i,j).

	:param i: First coordinate of the pixel.
	:type i: int

	:param j: Second coordinate of the pixel.
	:type j: int

	:param ii: Integral image.
	:type ii: ndarray

	:param sz_area: Number of surrounding neighbours to compute density.
	:type sz_area: int

	:return: int
*/
int Density::density_pix(int i, int j, const Mat* ii, int sz_area)
{
	int m, n;
	m = ii->rows;
	n = ii->cols;
	int x0 = max(0, i - sz_area);
	int	y0 = max(0, j - sz_area);
	int	x1 = min(m - 1, i + sz_area);
	int	y1 = min(n - 1, j + sz_area);
	return 255 - integr_api->integrate(ii, x0, y0, x1, y1) / float((x1 - x0)*(y1 - y0));
	/*
		m, n = ii.shape
		x0 = max(0, i - sz_area)
		y0 = max(0, j - sz_area)
		x1 = min(m - 1, i + sz_area)
		y1 = min(n - 1, j + sz_area)
		return 255 - integrate(ii, x0, y0, x1, y1) / float((x1 - x0)*(y1 - y0))
	*/
}

/*
"""
This function computes the density image of an image.
Computing the intensity of a pixel in base of the average
of its surrounding pixels.

:param img: Image to use.
:type img: ndarray

:param radius: Radius of the neighbours to take into account.
:type radius: int

:return: ndarray
"""
*/
Mat Density::density_img(const Mat* img, int radius, Config* cfg)
{
	int m = img->rows;
	int n = img->cols;


	Mat ii = integr_api->integral_image(img);

	if(cfg!=NULL && cfg->verboseDens)
		cfg->writeMatToFile("IntegralImage",&ii);

	Mat dens_img = Mat(m, n, CV_32F);
	
	for (int i = 0; i < m; i++) {
		int x0 = max(0, i - radius);
		int x1 = min(m - 1, i + radius);
		for (int j = 0; j < n; j++){
			int y0 = max(0, j - radius);
			int y1 = min(n - 1, j + radius);

			dens_img.at<float>(i, j) = 255 - (integr_api->integrate(&ii, x0, y0, x1, y1)) / float((x1 - x0)*(y1 - y0));
		}
	}

	/*
	m, n = img.shape
    ii = integral_image(np.asarray(img))

    dens_img = np.empty((m, n))
    for i in range(m):
        x0 = max(0, i - radius)
        x1 = min(m - 1, i + radius)
        for j in range(n):
            y0 = max(0, j - radius)
            y1 = min(n - 1, j + radius)
            dens_img[i, j] = 255 - integrate(ii, x0, y0, x1, y1) / float((x1-x0)*(y1-y0))

    return dens_img
	*/
	return dens_img;
}

