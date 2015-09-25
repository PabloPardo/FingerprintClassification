#include "MainRF.h"
#include "utils.h"
#include "opencv2\highgui\highgui.hpp"
#include "LearningRF.h"
#include <time.h>
#include <fstream>


using namespace cv;

MainRF::MainRF()
{
	learner = new LearningRF();
	learner->prop->verbose = true;
}

MainRF::~MainRF()
{
	delete learner;
}

void MainRF::Extraction(const char* labelsAndFeaturesPath, const char* imagesPath,const char* outPath)
{
	LabelsAndFeaturesData lfData = readCSV(labelsAndFeaturesPath);
	
	vector<Mat*>* imgData = new vector<Mat*>();
	for (unsigned int i = 0; i < lfData.imgFileNames.size(); i++)
	{
		clock_t time_a = clock();
		Mat featurei = lfData.features.row(i);
		Mat* in = new Mat(imread(imagesPath + lfData.imgFileNames[i], IMREAD_GRAYSCALE));
		imgData->push_back(in);
	}
	Mat rawFeaturesWithoutNFIQ;
	learner->Extract(imgData, &rawFeaturesWithoutNFIQ);

	Mat rawFeatures = Mat(rawFeaturesWithoutNFIQ.rows, rawFeaturesWithoutNFIQ.cols + lfData.features.cols, CV_32F);

	for (int i = 0; i < rawFeatures.rows; i++)
	{
		Mat rowi = rawFeaturesWithoutNFIQ.row(i);
		hconcat(rowi, lfData.features.row(i), rowi);
		rowi.copyTo(rawFeatures.row(i));
	}
	
	exportFileFeatures(rawFeatures, lfData.imgFileNames, ((std::string)outPath + "/RawData.csv").c_str());
}

void MainRF::NormalizeFitAndPredict(TrainPaths tPaths, PredictPaths pPaths, const char* results)
{
	LearnData result;
	//result.normalization = importNormalization(tPaths.normalizationFile);
	LabelsAndFeaturesData lfTData = readCSV(tPaths.labelsPath);	
	Mat trainData = importFileFeatures(tPaths.dataPath,false,Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	learner->CreateNorm(&trainData, &(result.normalization), &normTrainData);
	
	learner->Fit(&(result.rtrees), &(lfTData.matrix), &normTrainData);
	
	LabelsAndFeaturesData lfPData = readCSV(pPaths.labelsPath);	
	Mat predictData = importFileFeatures(pPaths.dataPath,false,Constants::TOTAL_FEATURES);
	Mat normPredictData;
	learner->Normalize(&predictData,&normPredictData,&(result.normalization));
		
	ofstream file;
	file.open(results);

	for(int i = 0; i < normPredictData.rows; i++)
	{
		float *probs;
		Mat* rowi = new Mat(normPredictData.row(i));
		learner->Predict(&probs,result.rtrees,rowi);
		file << lfPData.imgFileNames[i];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << probs[j];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << lfPData.matrix.at<int>(i,j);
		file << endl;
	}
	file.close();
}
