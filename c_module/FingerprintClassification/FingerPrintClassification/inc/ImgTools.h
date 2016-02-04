#include "opencv2\core\core.hpp"

using namespace cv;
#pragma once
struct ImgProcessingProperties
{
	int n_bins; //	Number of histogram bins.
	int rad_grad; // Gradient Radius.
	int rad_dens; // Density Radius.
	int rad_entr; // Entropy Radius.
	
	ImgProcessingProperties() {
		n_bins = 32;
		rad_grad = 1;
		rad_dens = 3;
		rad_entr = 5;
	};
	friend std::ostream& operator<<(std::ostream& os, const ImgProcessingProperties& prop);
};

class ImgTools
{
	void CropImage(Mat*, int, int, const Mat);
	void GetImageRegions(Mat***, const Mat);
public:
	ImgProcessingProperties* prop;
	ImgTools();
	~ImgTools();
	void ImageExtraction(const Mat, Mat*);
	void Extract(const vector<string>, Mat*);
	void ImportImageFeatures(vector<string>*, Mat*, const char*, const int);
	void ExportImageFeatures(Mat, vector<string>, const char*);
};