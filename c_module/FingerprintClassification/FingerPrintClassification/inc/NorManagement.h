#include "opencv2\core\core.hpp"

using namespace cv;
#pragma once
class NorManagement
{
public:
	static void Normalize(const Mat, Mat*, const Mat);
	static void CreateNorm(const Mat, Mat*, Mat* output = NULL);
	static void LoadNormalization(Mat*, const char*, int);
	static void SaveNormalization(const Mat, const char*);
};