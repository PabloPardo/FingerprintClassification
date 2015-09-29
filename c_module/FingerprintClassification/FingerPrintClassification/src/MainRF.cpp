#include "MainRF.h"
#include "utils.h"
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

void concatMatrix(const Mat in,const Mat c, Mat* out)
{
	Mat ret = Mat(in.rows, in.cols + c.cols, CV_32F);

	for (int i = 0; i < ret.rows; i++)
	{
		Mat rowi = in.row(i);
		hconcat(rowi, c.row(i), rowi);
		rowi.copyTo(ret.row(i));
	}
	*out = ret;
}

void MainRF::Extraction(const char* labelsAndFeaturesPath, const char* imagesPath, const char* outPath)
{
	LabelsAndFeaturesData lfData = readCSV(labelsAndFeaturesPath, imagesPath);
	
	Mat rawFeaturesWithoutNFIQ;
	learner->Extract(lfData.imgPaths, lfData.imgFileNames, &rawFeaturesWithoutNFIQ);

	Mat rawFeatures;

	concatMatrix(rawFeaturesWithoutNFIQ, lfData.features, &rawFeatures);
	
	exportFileFeatures(rawFeatures, lfData.imgFileNames, outPath);
}

void MainRF::ExtractFingerPrint(float** features, unsigned char* img, int w, int h, float* nfiqFeatures)
{
	Mat in = Mat(w, h, CV_8U, img);
	vector<Mat*>* imgData = new vector<Mat*>();
	
	Mat rawFeatures;
	learner->ImageExtraction(in, &rawFeatures);
	Mat nfiq = Mat(1,13,CV_32F,nfiqFeatures);
	Mat ret;
	concatMatrix(rawFeatures,nfiq,&ret);

	*features = (float*)ret.data;
}
/*
void MainRF::NormalizeFitAndPredict(TrainPaths tPaths, PredictPaths pPaths, const char* results)
{
	LearnData result;
	LabelsAndFeaturesData lfTData = readCSV(tPaths.labelsPath);	
	Mat trainData = importFileFeatures(tPaths.dataPath,false,Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	learner->CreateNorm(&trainData, &(result.normalization), &normTrainData);
	
	learner->Fit(&(result.rtrees), &(lfTData.matrix), &normTrainData);
	
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char fileName[10000];
		sprintf_s(fileName, "%smodel_%d.xml", "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/", i);
		result.rtrees[i].save(fileName);
	}
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	for(int i=0; i<Constants::NUM_CLASSIFIERS; i++)
	{
		// Load the trained model
		char modelName[10000];
		sprintf_s(modelName,"%smodel_%i.xml", "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/", i);
		models[i].load(modelName);
		
	}

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
		learner->Predict(&probs,models,rowi);
		file << lfPData.imgFileNames[i];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << probs[j];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << lfPData.matrix.at<int>(i,j);
		file << endl;
	}
	file.close();
}
*/
void MainRF::Fit(TrainPaths tPaths, const char* outputDir)
{
	LearnData result;
	LabelsAndFeaturesData lfTData = readCSV(tPaths.labelsPath);	
	Mat trainData = importFileFeatures(tPaths.dataPath,false,Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	learner->CreateNorm(trainData, &(result.normalization), &normTrainData);
	
	learner->Fit(&(result.rtrees), lfTData.matrix, normTrainData);
	
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char fileName[10000];
		sprintf_s(fileName, "%smodel_%d.xml", outputDir, i);
		result.rtrees[i].save(fileName);
	}

	saveNormalization(result.normalization, ((string)outputDir + "/normalization.csv").c_str());
}

void MainRF::PredictTest(PredictPaths pPaths, const char* results)
{
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		// Load the trained model
		char modelName[10000];
		sprintf_s(modelName,"%smodel_%i.xml", pPaths.modelDir, i);
		models[i].load(modelName);
	}

	Mat norMat;
	loadNormalization(&norMat,((string)pPaths.modelDir + "/normalization.csv").c_str());

	LabelsAndFeaturesData lfPData = readCSV(pPaths.labelsPath);	
	Mat predictData = importFileFeatures(pPaths.dataPath, false, Constants::TOTAL_FEATURES);
	Mat normPredictData;
	learner->Normalize(predictData, &normPredictData, norMat);
		
	ofstream file;
	file.open(results);

	for(int i = 0; i < normPredictData.rows; i++)
	{
		float *probs;
		Mat rowi = Mat(normPredictData.row(i));
		learner->Predict(&probs, models, rowi);
		file << lfPData.imgFileNames[i];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << probs[j];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << lfPData.matrix.at<int>(i,j);
		file << endl;
	}
	file.close();
}

void MainRF::Predict(float** probs, PredictPaths pPaths, float* features)
{
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char modelName[10000];
		sprintf_s(modelName,"%smodel_%i.xml", pPaths.modelDir, i);
		models[i].load(modelName);
	}

	Mat norMat;
	loadNormalization(&norMat,((string)pPaths.modelDir + "/normalization.csv").c_str());

	Mat predictData = Mat(1,Constants::TOTAL_FEATURES,CV_32F, features);
	Mat normPredictData;
	learner->Normalize(predictData,&normPredictData,norMat);
	learner->Predict(probs,models,normPredictData);
}