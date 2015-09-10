#include "opencv2\core\core.hpp"

#ifndef FPFEATURESLIBRARY_H
#define	FPFEATURESLIBRARY_H

struct Config {
	bool verboseGrad;
	bool verboseHough;
	bool verboseDens;
	bool verboseEntropy;
	bool verboseDiff;
	char* path;
	char* fileName;
};


#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) char* getVersion();
	__declspec(dllimport) void setConfig(Config* cfg);
	__declspec(dllimport) cv::Mat hist_density(const cv::Mat* image, int radius, int n_bins);
	__declspec(dllimport) cv::Mat hist_grad(const cv::Mat* image, int radius, int n_bins);
	__declspec(dllimport) cv::Mat diferentiate_img(const cv::Mat* image);
	__declspec(dllimport) cv::Mat hist_entropy(const cv::Mat* img, int radius, int n_bins);
	__declspec(dllimport) cv::Mat hist_hough(const cv::Mat* img, int n_bins);
#ifdef __cplusplus
}
#endif

#endif  // FPFEATURESLIBRARY_H