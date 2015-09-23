#include <string>

#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "FingerPrintClassification.h"

#ifndef UTILS_H
#define UTILS_H

void throwError(std::string error);
int countLines(const char *path);

struct Constants
{
	static const int TOTAL_FEATURES = 914;

	static const int NUM_ROW_SEGMENTS = 3;
	static const int NUM_COL_SEGMENTS = 2;
	static const int NUM_CLASSIFIERS = 6;
	static const int NUM_FEATURES = 13;
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

LabelsAndFeaturesData readCSV(const char*);

cv::Mat oneVsAll(cv::Mat labels, int tar_class);

void printParamsRF(const Properties& prop);

void printParamsSVM(const PropertiesSVM& propSVM);

cv::Mat importFileFeatures(const char*, bool, const int);

void exportFileFeatures(cv::Mat, std::vector<std::string>,const char*);

cv::Mat createNormalizationFile(const char* outPath, cv::Mat trainSamples);

cv::Mat readTrainedMeanStd(const char* normalizationFilePath,cv::Mat sample);

void allocateRtrees(CvRTrees***, const int, const int);

void allocateSVMs(CvSVM***, const int, const int);

void releaseRTrees(CvRTrees**, const int, const int);

void releaseSVMs(CvSVM**, const int, const int);

#endif /* UTILS_H */