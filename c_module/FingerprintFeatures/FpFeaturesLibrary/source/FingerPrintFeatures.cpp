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


Mat FingerPrintFeatures::hist_density(const Mat* img, int radius, int n_bins=64)
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
	float lranges[] = { minVal , maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&dens_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		return hist.reshape(0,1);
	}
	else
	{
		return Mat(1,32,CV_32F,0.);
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
Mat FingerPrintFeatures::hist_grad(const Mat* img, int radius, int n_bins)
{	
	Mat grad_img = grad_api->gradient_img(img, radius,cfg);
	
	Mat grad_flatten = grad_img.reshape(0, 1);

	double minVal, maxVal;
	minMaxLoc(grad_flatten, &minVal, &maxVal);
	// Set histogram bins count
	int histSize[] = { n_bins };
	// Set ranges for histogram bins
	float lranges[] = { minVal, maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(cfg->verboseGrad)
		cfg->writeMatToFile("GradImage", &grad_flatten);
	Mat ret;
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&grad_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		ret = Mat(hist.reshape(0,1));
	}
	else
	{
		ret = Mat(1,32,CV_32F,0.);
	}	
	return ret;
}

Mat FingerPrintFeatures::diferentiate_img(const Mat* img)
{
	int m, n;
	m = img->rows;
	n = img->cols;
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
		int int1 = api->integrate(&ii, x0, y0, x1, y1);
		int int2 = api->integrate(&ii, x2, y2, x3, y3);
		dif.at<float>(0,i) = (abs(int1 - int2));
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
		int int1 = api->integrate(&ii, x0, y0, x1, y1);
		int int2 = api->integrate(&ii, x2, y2, x3, y3);

		dif.at<float>(0,i+3) = (abs(int1 - int2));
	}
	if (cfg->verboseDiff)
		cfg->writeMatToFile("DiffImage", &dif);
	return dif;

	/*
	m, n = img.shape

	row_split = m / 3
	col_split = n / 2

	ii = integral_image(img)

	dif = []
	for i in range(3) :
		x0 = i*row_split
		y0 = 0
		x1 = (i + 1)*row_split - 1
		y1 = col_split

		x2 = x0
		y2 = col_split
		x3 = x1
		y3 = n - 1
		int1 = integrate(ii, x0, y0, x1, y1)
		int2 = integrate(ii, x2, y2, x3, y3)

		dif.append(np.abs(int1 - int2))

		row_split = m / 2

		for i in range(2) :
			x0 = 0
			y0 = i*col_split
			x1 = row_split
			y1 = (i + 1)*col_split - 1

			x2 = row_split
			y2 = y0
			x3 = m - 1
			y3 = y1
			int1 = integrate(ii, x0, y0, x1, y1)
			int2 = integrate(ii, x2, y2, x3, y3)

			dif.append(np.abs(int1 - int2))

			return dif
			*/
}



void FingerPrintFeatures::initLogTable(int r)
{
	int total = 0;
	int d = r*2;
	for(int i = (r) + 1; i <= d; i++)
	{
		for(int j = (r) + 1; j <= d; j++)
		{
			if(i>=j)
				total += (i*j)+1;		
		}	
	}

	int next = 0;
	for(int i = (r) + 1; i <= d; i++)
	{
		for(int j = (r) + 1; j <= d; j++)
		{
			if(i>=j)
			{
				int n = i*j;
				for(int k=0;k<=n;k++)
				{
					char buffer[100] = {0};
					int number_base = 10;
					float key = k/(float)n;
					float logKey = log(key);
					std::string key_str = _itoa(k,buffer,number_base);
					key_str += _itoa(n,buffer,number_base);
					logTable[atoi(key_str.c_str())] = logKey;
				}
			}
		}	
	}
	
}

void logMeu(cv::Mat* in,int den,cv::Mat** out, float logTable[])
{
	(*out) = new Mat(in->rows,in->cols,CV_32F);
	
	for(int i=0;i<in->rows;i++)
	{
		for(int j=0;j<in->cols;j++)
		{
			(*out)->at<float>(i,j) = 0.;
			/*
			int numerador = (int)in->at<float>(i,j);
			char buffer[100] = {0};
			std::string key_str = (std::string)_itoa(numerador,buffer,10) + (std::string)_itoa(den,buffer,10);
			(*out)->at<float>(i,j) = logTable[atoi(key_str.c_str())];
			*/
		}
	}
}

/*************************************45 seconds!!!!*****************************************
double log2( double number ) {
   return log(number)/log(2.0f);
}
 
float FingerPrintFeatures::entropy(cv::Mat* input) {
   std::map<uchar , int> frequencies;
   for(int i = 0; i < input->rows; i++)
	   for(int j = 0; j < input->cols; j++)
			frequencies[input->data[i*input->cols+j]]++;
   int numlen = input->rows*input->cols;
   double infocontent = 0 ;
   for each ( std::pair<uchar , int> p in frequencies ) {
      double freq = static_cast<double>( p.second ) / numlen ;
	  char buffer[100] = {0};
	  std::string key_str = (std::string)itoa(p.second,buffer,10) + (std::string)itoa(numlen,buffer,10);
	  infocontent += freq * logTable[atoi(key_str.c_str())];// log2( freq ) ;
   }
   infocontent *= -1 ;
   return infocontent;
}
*************************************************************************************************/
/***
* Calculates the entropy of a matrix as a single float value *
*/
float FingerPrintFeatures::entropy(const cv::Mat* input,const cv::Mat* disk) {
	
	float frequencies[256] = {0};
	for(int i = 0; i < input->rows; i++)
		for(int j = 0; j < input->cols; j++)
			frequencies[input->at<unsigned char>(i,j)]++;
	
	cv::Mat hist(1,256,CV_32F,frequencies);
    //std::cout << hist << std::endl;
	hist /= input->total();
	//std::cout << hist << std::endl;
 	cv::Mat logP;
	//logMeu(&hist,input->rows*input->cols,&logP,logTable);
	cv::log(hist,logP);
	logP /= 0.6931471805599453;
	//std::cout << logP << std::endl;

    float entropy = -1*cv::sum(hist.mul(logP)).val[0];

	return entropy;
}
/*
float FingerPrintFeatures::entropy(cv::Mat* input, cv::Mat* disk) {
	/// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    /// Compute the histograms:
    cv::Mat hist;
	//std::cout << *input << endl << endl;
	calcHist( input, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    //std::cout << hist << endl << endl;
	hist /= input->total();
	//std::cout << hist << endl << endl;
    cv::Mat logP;
    //logMeu(&hist,logKeys,logValues,&logP);
	cv::log(hist,logP);
	//std::cout << hist << endl;

    float entropy = -1*cv::sum(hist.mul(logP)).val[0];

    return entropy;
}
*/

Mat getSubSetNeighbours(const Mat* in,int row,int col, int radius)
{
	int start_row = cv::max(0, row - radius);
	int end_row = cv::min(in->rows, row + radius + 1);
	int start_col = cv::max(0, col - radius);
	int end_col = cv::min(in->cols, col + radius + 1);
	
	cv::Rect rect = cv::Rect(start_col,start_row,end_col-start_col,end_row-start_row);

	return Mat((*in)(rect));
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
Mat FingerPrintFeatures::hist_entropy(const Mat* img, int radius, int n_bins) {
	
	Mat disk = Mat(cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(radius,radius)));
	initLogTable(radius);


	Mat output = Mat(img->rows,img->cols,CV_32F);

	for(int i=0; i<img->rows;i++)
	{
		for(int j=0; j<img->cols;j++)
		{
			try
			{
			Mat subSet = getSubSetNeighbours(img,i,j,(radius));
			output.at<float>(i,j) = entropy(&subSet,&disk);

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
	float lranges[] = { minVal, maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0))
	{
		calcHist(&output_flatten, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		return hist.reshape(0,1);
	}
	else
	{
		return Mat(1,32,CV_32F,0.);
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
Mat FingerPrintFeatures::hist_hough(const Mat* img, int n_bins) {
	int threshold = 10;
	int line_length = 8;
	int line_gap = 3;
	Mat edges;
	
	GaussianBlur(*img, edges, Size(9,9), 2.);
	
	/*if (cfg->verboseHough)
		cfg->writeMatToFile("blur", &edges);*/

	Canny(edges, edges, 1, 25, 3);
	
	if(cfg->verboseHough)
		cfg->writeMatToFile("edges",&edges);
	
	cv::Mat lines;
	
	HoughLinesP(edges, lines, 1, CV_PI / 100, threshold, line_length, line_gap);
	
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
	float lranges[] = { minVal, maxVal };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };
	if(!(minVal == 0 && maxVal == 0) && magnitudes.cols >= 32)
	{
		calcHist(&magnitudes, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
		return hist.reshape(0,1);
	}
	else
	{
		return cv::Mat(1,32,CV_32F,0.);
	}
}
