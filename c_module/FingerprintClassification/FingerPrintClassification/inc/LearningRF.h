#include "opencv2\ml\ml.hpp"
#include "utils.h"
#pragma once

using namespace cv;
using namespace std;

struct LearnData
{
	CvRTrees* rtrees;
	cv::Mat normalization;
	void write(const char* outPath);
};

class LearningRF
{
	void ImageExtraction(const Mat*,Mat*);
public:
	Properties* prop;
	LearningRF();
	~LearningRF();
	void Extract(const vector<Mat*>*, Mat*);
	void Fit(CvRTrees**, const Mat*, const Mat*);
	void Predict(float**, CvRTrees*, const Mat*);
	void Normalize(const Mat*, Mat*, const Mat*);
	void CreateNorm(const Mat*, Mat*, Mat* output = NULL);
};

