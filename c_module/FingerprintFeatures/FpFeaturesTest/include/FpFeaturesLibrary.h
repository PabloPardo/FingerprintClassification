#include "opencv2\core\core.hpp"

#ifndef FPFEATURESLIBRARY_H
#define	FPFEATURESLIBRARY_H

using namespace cv;
struct Config {
	bool verboseGrad;
	bool verboseHough;
	bool verboseDens;
	bool verboseEntropy;
	char* path;
	char* fileName;
};


#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) char* getVersion();
	__declspec(dllimport) void setConfig(Config*);
	__declspec(dllimport) void hist_density(Mat*, const Mat*, int, int);
	__declspec(dllimport) void hist_grad(Mat*, const Mat* image, int, int);
	__declspec(dllimport) void diferentiate_img(Mat*, const Mat*);
	__declspec(dllimport) void hist_entropy(Mat*, const Mat*, int, int);
	__declspec(dllimport) void hist_hough(Mat*, const Mat*, int);
#ifdef __cplusplus
}
#endif

#endif  // FPFEATURESLIBRARY_H