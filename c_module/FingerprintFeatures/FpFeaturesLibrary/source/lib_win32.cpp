#include "opencv2\core\core.hpp"
#include "FingerPrintFeatures.h"

using namespace std;
using namespace cv;

FingerPrintFeatures* api = new FingerPrintFeatures(new Config());

extern "C" __declspec(dllexport) char* getVersion() {
	return "2.0";
}

extern "C" __declspec(dllexport) void setConfig(Config* cfg) {
	api->cfg = cfg;
}

extern "C" __declspec(dllexport) Mat hist_density(const Mat* image, int radius, int n_bins) {
	return api->hist_density(image, radius, n_bins);
}
extern "C" __declspec(dllexport) Mat hist_grad(const Mat* image, int radius, int n_bins) {
	return api->hist_grad(image, radius, n_bins);
}
extern "C" __declspec(dllexport) Mat diferentiate_img(const Mat* image) {
	return api->diferentiate_img(image);
}
extern "C" __declspec(dllexport) Mat hist_entropy(Mat* img, int radius = 5, int n_bins = 64) {
	return api->hist_entropy(img, radius, n_bins);
}
extern "C" __declspec(dllexport) Mat hist_hough(Mat* img, int n_bins) {
	return api->hist_hough(img,n_bins);
}