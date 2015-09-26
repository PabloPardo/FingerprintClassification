#include "opencv2\core\core.hpp"
#include "FingerPrintFeatures.h"

using namespace std;
using namespace cv;

FingerPrintFeatures* api = new FingerPrintFeatures(new Config());
#ifdef __cplusplus
extern "C" {
#endif
	
	__declspec(dllexport) void setConfig(Config* cfg) {
		api->cfg = cfg;
	}
	
	__declspec(dllexport) void hist_density(Mat* ret, const Mat* image, int radius, int n_bins) {
		api->hist_density(ret, image, radius, n_bins);
	}

	__declspec(dllexport) void hist_grad(Mat* ret, const Mat* image, int radius, int n_bins) {
		api->hist_grad(ret, image, radius, n_bins);
	}

	__declspec(dllexport) void diferentiate_img(Mat* ret, const Mat* image) {
		api->diferentiate_img(ret, image);
	}

	__declspec(dllexport) void hist_entropy(Mat* ret, const Mat* img, int radius = 5, int n_bins = 64) {
		api->hist_entropy(ret, img, radius, n_bins);
	}
	
	__declspec(dllexport) void hist_hough(Mat* ret, const Mat* img, int n_bins) {
		api->hist_hough(ret, img,n_bins);
	}

	__declspec(dllexport) float entropy(const Mat* input, const Mat* shape) {
		return api->entropy(input,shape);
	}
#ifdef __cplusplus
}
#endif