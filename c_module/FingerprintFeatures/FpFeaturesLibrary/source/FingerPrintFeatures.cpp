#include "FingerPrintFeatures.h"
#include "Gradient.h"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "types.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <cmath>


using namespace std;
FingerPrintFeatures::FingerPrintFeatures(Config* cfgValue)
{
	grad_api = new Gradient();
	dens_api = new Density();
	cfg = cfgValue;
}

FingerPrintFeatures::~FingerPrintFeatures(void)
{
	grad_api->~Gradient();
	dens_api->~Density();
	delete grad_api;
	delete dens_api;
}
/*
"""
Computes the histogram of densities from an image.

: param img : Image to be precessed.
: type img : ndarray

: param radius : Radius to use for the density calculation.
: type radius : int

: param n_bins : Number of bins to compute the histogram.
: type n_bins : int

: return : list of int
"""
*/


void FingerPrintFeatures::hist_density(Mat* ret, const Mat img, int radius, int n_bins)
{	
	Mat dens_img = dens_api->density_img(img, radius,cfg);
	
	if(cfg->verboseDens)
		cfg->writeMatToFile("DensityImage", &dens_img);
	
	Mat dens_flatten = dens_img.reshape(0, 1);
	double minVal, maxVal;
	minMaxLoc(dens_flatten, &minVal, &maxVal);

	// Set histogram bins count
	int histSize[] = { n_bins };
	// Set ranges for histogram bins
	float lranges[] = { (float)minVal , (float)maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&dens_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		*ret = hist.reshape(0,1);
	}
	else
	{
		*ret = Mat(1,32,CV_32F,0.);
	}
}

/*
"""
Computes the histogram of gradients from an image.

: param img : Image to be precessed.
: type img : ndarray

: param radius : Radius to use for the gradient calculation.
: type radius : int

: param n_bins : Number of bins to compute the histogram.
: type n_bins : int

: return : list of int
"""
*/
void FingerPrintFeatures::hist_grad(Mat* ret, const Mat img, int radius, int n_bins)
{	
	Mat grad_img = grad_api->gradient_img(img, radius, cfg);
	
	Mat grad_flatten = grad_img.reshape(0, 1);

	double minVal, maxVal;
	minMaxLoc(grad_flatten, &minVal, &maxVal);
	// Set histogram bins count
	int histSize[] = { n_bins };
	// Set ranges for histogram bins
	float lranges[] = { (float)minVal , (float)maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(cfg->verboseGrad)
		cfg->writeMatToFile("GradImage", &grad_flatten);
	
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&grad_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		*ret = Mat(hist.reshape(0,1));
	}
	else
	{
		*ret = Mat(1,32,CV_32F,0.);
	}	
	
}

void FingerPrintFeatures::diferentiate_img(Mat* ret, const Mat img)
{
	int m, n;
	m = img.rows;
	n = img.cols;
	Integral* api = new Integral();

	int row_split = m / 3;
	int col_split = n / 2;

	Mat ii = api->integral_image(img);

	Mat dif = Mat(1,5,CV_32F);
	for (int i = 0; i < 3; i++) {
		int x0 = i*row_split;
		int y0 = 0;
		int x1 = (i + 1)*row_split - 1;
		int y1 = col_split;

		int x2 = x0;
		int y2 = col_split;
		int x3 = x1;
		int y3 = n - 1;
		int int1 = api->integrate(ii, x0, y0, x1, y1);
		int int2 = api->integrate(ii, x2, y2, x3, y3);
		dif.at<float>(0,i) = (float)(abs(int1 - int2));
	}
	row_split = m / 2;

	for (int i = 0; i < 2; i++){
		int x0 = 0;
		int y0 = i*col_split;
		int x1 = row_split;
		int y1 = (i + 1)*col_split - 1;

		int x2 = row_split;
		int y2 = y0;
		int x3 = m - 1;
		int y3 = y1;
		int int1 = api->integrate(ii, x0, y0, x1, y1);
		int int2 = api->integrate(ii, x2, y2, x3, y3);

		dif.at<float>(0,i+3) = (float)(abs(int1 - int2));
	}
	if (cfg->verboseDiff)
		cfg->writeMatToFile("DiffImage", &dif);
	*ret = dif;
}

float FingerPrintFeatures::entropy(const Mat input,const Mat disk) {
	
	float frequencies[256] = {0};
	int nDisk = 0;
	for(int i = 0; i < input.rows; i++)
		for(int j = 0; j < input.cols; j++)
		{
			if(disk.at<unsigned char>(i,j) == 1)
			{
				frequencies[input.at<unsigned char>(i,j)]++;
				nDisk++;
			}
		}
	
	cv::Mat hist(1,256,CV_32F,frequencies);
    //std::cout << hist << std::endl;
	hist /= nDisk;
	//std::cout << hist << std::endl;
 	cv::Mat logP;
	//logMeu(&hist,input->rows*input->cols,&logP,logTable);
	cv::log(hist,logP);
	logP /= 0.6931471805599453;
	//std::cout << logP << std::endl;

    float entropy = (float)(-1*cv::sum(hist.mul(logP)).val[0]);

	return entropy;
}

Mat getSubSetNeighbours(const Mat in,int row,int col, int radius)
{
	int start_row = cv::max(0, row - radius);
	int end_row = cv::min(in.rows, row + radius + 1);
	int start_col = cv::max(0, col - radius);
	int end_col = cv::min(in.cols, col + radius + 1);
	
	cv::Rect rect = cv::Rect(start_col,start_row,end_col-start_col,end_row-start_row);

	return Mat((in)(rect));
}
Mat getDiskSubSet(const Mat disk, int i, int j, int rows, int cols, int radius)
{
	int row_l = i - radius;
	int row_r = i + radius;
	int col_l = j - radius;
	int col_r = j + radius;

	int start_row = cv::max(0, row_l) - row_l;
	int end_row = cv::min(rows, row_r + 1) - row_l;
	int start_col = cv::max(0, col_l) - col_l;
	int end_col = cv::min(cols, col_r + 1) - col_l;
	cv::Rect subDisk = cv::Rect(start_col,start_row,end_col-start_col,end_row-start_row);

	return Mat((disk)(subDisk));
}
/*
"""
Computes the histogram of entropies from an image.

: param img : Image to be precessed.
: type img : ndarray

: param radius : Radius to use for the entropy calculation.
: type radius : int

: param n_bins : Number of bins to compute the histogram.
: type n_bins : int

: return : list of int
"""
*/
void FingerPrintFeatures::hist_entropy(Mat* ret, const Mat img, int radius, int n_bins) {
	
	Mat disk = Mat(cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(radius*2+1,radius*2+1)));
	
	//initLogTable(radius);

	Mat output = Mat(img.rows,img.cols,CV_32F);

	for(int i=0; i<img.rows;i++)
	{
		for(int j=0; j<img.cols;j++)
		{
			try
			{
			Mat subSet = getSubSetNeighbours(img,i,j,(radius));
			Mat subDisk = getDiskSubSet((const Mat)disk,i,j,img.rows,img.cols,(radius));
			output.at<float>(i,j) = entropy(subSet,subDisk);

			//output->at<float>(i,j) = (i+j)/(float)img->rows*img->cols;
			} catch (...) {
				std::cout << "Error en " << i << "," << j << std::endl;
				throw;
			}
		}
	}

	//cout << *output << endl;
	if(cfg->verboseEntropy)
		cfg->writeMatToFile("Entropy",&output);
	
	Mat output_flatten = output.reshape(0, 1);
	double minVal, maxVal;
	minMaxLoc(output_flatten, &minVal, &maxVal);
	// Set histogram bins count
	int histSize[] = { n_bins };
	// Set ranges for histogram bins
	float lranges[] = { (float)minVal , (float)maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&output_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		*ret = hist.reshape(0,1);
	}
	else
	{
		*ret = Mat(1,32,CV_32F,0.);
	}
}


/*
"""
Computes the histogram of hough line magnitudes.

: param img : Image
: type img : nparray

: param n_bins : Number of bins
: type n_bins : int

: return : histogram of hough lines magnitudes
"""
*/
void FingerPrintFeatures::hist_hough(Mat* ret, const Mat img, int n_bins) {
	int threshold = 10;
	int line_length = 8;
	int line_gap = 3;
	Mat edges;
	
	GaussianBlur(img, edges, Size(17,17), 2.);
	
	if (cfg->verboseHough)
		cfg->writeMatToFile("blur", &edges);

	Canny(edges, edges, 1, 25, 3);
	
	if(cfg->verboseHough)
		cfg->writeMatToFile("edges",&edges);
	
	cv::Mat lines;
	
	HoughLinesP(edges, lines, 1, CV_PI / 100, threshold, line_length, line_gap);
	
	if(cfg->verboseHough)
		cfg->writeMatToFile("lines", &lines);

	Mat magnitudes = Mat(1, lines.cols, CV_32F);
	for (int i = 0; i < lines.cols; i++)
	{
		Vec4i l = lines.at<Vec4i>(0,i);
		magnitudes.at<float>(0,i) = (float)norm(l);
	}
	
	if(cfg->verboseHough)
		cfg->writeMatToFile("Magnitudes",&magnitudes);
	
	//Mat magnitudes_flatten = magnitudes.reshape(0, 1);
	double minVal, maxVal;
	minMaxLoc(magnitudes, &minVal, &maxVal);
	// Set histogram bins count
	int histSize[] = { n_bins };
	// Set ranges for histogram bins
	float lranges[] = { (float)minVal , (float)maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0) && magnitudes.cols >= 32)
	{
		calcHist(&magnitudes, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		*ret = hist.reshape(0,1);
	}
	else
	{
		*ret = cv::Mat(1,32,CV_32F,0.);
	}
}
