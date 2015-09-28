#include <string>

#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

#ifndef UTILS_H
#define UTILS_H

void throwError(std::string error);
int countLines(const char *path);

using namespace cv;

struct Constants
{
	static const int TOTAL_FEATURES = 923;

	static const int NUM_ROW_SEGMENTS = 3;
	static const int NUM_COL_SEGMENTS = 2;
	static const int NUM_CLASSIFIERS = 6;
	static const int NUM_FEATURES = 22;
};

enum CSV_HEADERS 
{
	EmId=0,
	EmUsuari,
	EmNomFitxer,
	EmNumDit,
	EmBorrosa,
	EmPetita,
	EmNegre,
	EmClara,
	EmMotejada,
	EmDefectuosa,
	dedo,
	nfiq,
	foreground,
	numMinucias,
	uno,
	dos,
	tres,
	cuatro,
	cinco,
	seis,
	siete,
	ocho,
	nueve
};

struct LabelsAndFeaturesData {
	cv::Mat matrix;
	std::vector<std::string> imgFileNames;
	cv::Mat features;
};

struct Properties
{
	int n_bins; //	Number of histogram bins.
	int rad_grad; // Gradient Radius.
	int rad_dens; // Density Radius.
	int rad_entr; // Entropy Radius.
	int max_depth; // Max depth of the trees in the Random Forest.
	int min_samples_count; // Min samples needed to split a leaf.
	int max_categories; // Max number of categories.
	int max_num_of_trees_in_forest; // Max number of trees in the forest.
	int nactive_vars; // nactive_vars,

	bool verbose;

	Properties() {
		n_bins = 32;
		rad_grad = 1;
		rad_dens = 3;
		rad_entr = 5;
		max_depth = 16;
		min_samples_count = 2;
		max_categories = 3;
		max_num_of_trees_in_forest = 10;
		nactive_vars = 0;
		verbose = false;
	};
	friend std::ostream& operator<<(std::ostream& os, const Properties& prop);
};


LabelsAndFeaturesData readCSV(const char*);

cv::Mat CropImage(int, int, const cv::Mat*);

cv::Mat** GetImageRegions(const cv::Mat*);

cv::Mat oneVsAll(cv::Mat labels, int tar_class);

void printParamsRF(const Properties& prop);

void loadNormalization(Mat*, const char*);

void saveNormalization(const Mat*, const char*);

cv::Mat importFileFeatures(const char*, bool, const int);

void exportFileFeatures(cv::Mat, std::vector<std::string>,const char*);

cv::Mat createNormalizationFile(const char* outPath, cv::Mat trainSamples);

cv::Mat readTrainedMeanStd(const char* normalizationFilePath,cv::Mat sample);

void allocateRtrees(CvRTrees***, const int, const int);

void releaseRTrees(CvRTrees**, const int, const int);

#endif /* UTILS_H */