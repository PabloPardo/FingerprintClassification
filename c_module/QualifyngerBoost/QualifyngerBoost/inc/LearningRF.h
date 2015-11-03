#include "opencv2\ml\ml.hpp"
#include "utils.h"
#pragma once

using namespace cv;
using namespace std;

class LearningRF
{
public:
	Properties* prop;
	LearningRF();
	~LearningRF();
	void ImageExtraction(const Mat,Mat*);
	void Extract(const vector<string>, const vector<string>, Mat*);
	void Fit(CvRTrees**, const Mat, const Mat);
	void Predict(float**, CvRTrees*, const Mat);
	void Normalize(const Mat, Mat*, const Mat);
	void CreateNorm(const Mat, Mat*, Mat* output = NULL);
};

