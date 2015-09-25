#include "MainRF.h"
#include "utils.h"
#include "opencv2\highgui\highgui.hpp"
#include "LearningRF.h"
#include <time.h>


using namespace cv;

MainRF::MainRF()
{
	learner = new LearningRF();
}

MainRF::~MainRF()
{
	delete learner;
}

void MainRF::ExtractNormalizeAndFit(const char* labelsAndFeaturesPath, const char* imagesPath, const char* modelPath)
{
	LabelsAndFeaturesData lfData = readCSV(labelsAndFeaturesPath);
	
	vector<Mat*>* imgData = new vector<Mat*>();
	for (unsigned int i = 0; i < lfData.imgFileNames.size(); i++)
	{
		clock_t time_a = clock();
		Mat featurei = lfData.features.row(i);
		Mat in = imread(imagesPath + lfData.imgFileNames[i], IMREAD_GRAYSCALE);
		(*imgData)[i] = &in;
	}
	Mat rawFeaturesWithoutNFIQ;
	learner->Extract(imgData, &rawFeaturesWithoutNFIQ);
	Mat rawFeatures = Mat(rawFeaturesWithoutNFIQ.rows, rawFeaturesWithoutNFIQ.cols + lfData.features.cols, CV_64F);

	for (unsigned int i = 0; i < rawFeatures.rows; i++)
	{
		Mat rowi = rawFeaturesWithoutNFIQ.row(i);
		hconcat(rowi, lfData.features.row(i), rowi);
		rawFeatures.row(i) = rowi;
	}
	
	LearnData result;
	Mat normTS = Mat();

	learner->CreateNorm(&rawFeatures, &(result.normalization), &normTS);
	learner->Fit(&(result.rtrees), &(lfData.matrix), &normTS);

	result.write(modelPath);
}
