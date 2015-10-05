#include "MainRF.h"
#include "utils.h"
#include "LearningRF.h"
#include <time.h>
#include <fstream>
#include <io.h>
#include "opencv2\ml\ml.hpp"
#include "opencv2\core\core.hpp"

using namespace cv;

MainRF::MainRF()
{
	learner = new LearningRF();
	learner->prop->verbose = false;
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

void MainRF::ExtractFingerPrint(int* lenFeatures, float** features, unsigned char* img, int w, int h, float* nfiqFeatures)
{
	Mat in = Mat(w, h, CV_8U, img);
	Mat rawFeatures;
	learner->ImageExtraction(in, &rawFeatures);
	Mat nfiq = Mat(1,Constants::NUM_FEATURES, CV_32F, nfiqFeatures);
	Mat ret;
	concatMatrix(rawFeatures,nfiq,&ret);
	
	*lenFeatures = ret.cols;
	*features = new float[*lenFeatures];
	memcpy_s(*features, (*lenFeatures)*sizeof(float), (float*)ret.data, (*lenFeatures)*sizeof(float));
	/*
	for (int i = 0; i < *lenFeatures; i++)
		*features[i] = ret.at<float>(0, i);
		*/
}

void MainRF::Fit(const TrainPaths tPaths, const char* outputDir)
{
	Mat normalization;
	CvRTrees* rtrees;
	LabelsAndFeaturesData lfTData = readCSV(tPaths.labelsPath);	
	Mat trainData = importFileFeatures(tPaths.dataPath,false,Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	learner->CreateNorm(trainData, &normalization, &normTrainData);
	
	learner->Fit(&rtrees, lfTData.matrix, normTrainData);
	
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char fileName[10000];
		sprintf_s(fileName, "%smodel_%d.xml", outputDir, i);
		rtrees[i].save(fileName);
	}

	saveNormalization(normalization, ((string)outputDir + "/normalization.csv").c_str());
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

void MainRF::Predict(int* lenProbs, float** probs, Handle* hnd, float* features)
{
	CvRTrees* models = (CvRTrees*)hnd->rTrees;

	Mat* ptrMat = (Mat*)hnd->norMat;

	Mat nM = *ptrMat;
	
	Mat predictData = Mat(1,Constants::TOTAL_FEATURES,CV_32F, features);
	Mat normPredictData;
	learner->Normalize(predictData, &normPredictData, nM);
	learner->Predict(probs,models,normPredictData);
	*lenProbs = Constants::NUM_CLASSIFIERS;
}

void MainRF::InitModel(Handle** hnd, const char *modelPath)
{
	clock_t begin;
	double load_rtree = 0;
	Handle* ret = new Handle();
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	delete[] models;
	models = new CvRTrees[Constants::NUM_CLASSIFIERS];

	for (int i = 0; i < Constants::NUM_CLASSIFIERS; i++){
		// Initialice rtree;

		// Load the trained model
		char modelName[10000];
		sprintf_s(modelName, "%smodel_%i.xml", modelPath, i);
		if (_access_s(modelName, 0) != -1)
		{
			begin = clock();
			models[i].load(modelName);
			load_rtree += double(clock() - begin);
		}
		else
		{
			string err = "ERROR: file " + (string)modelName + " could not be opened. Is the path okay?";
			cerr << err << endl;
			throw exception(err.c_str());
		}
	}

	Mat nM;
	loadNormalization(&nM, ((string)modelPath + "/normalization.csv").c_str());

	ret->rTrees = models;
	ret->norMat = new Mat(nM);
	
	*hnd = ret;
	
	if (learner->prop->verbose)
		cout << "Load Trees[" << load_rtree << "]" << endl;
}

void MainRF::ReleaseModel(Handle* hnd)
{	
	delete (Mat*)hnd->norMat;
	delete[] (CvRTrees*)hnd->rTrees;
	delete (Handle*)hnd;
}

void MainRF::ReleaseFloatPointer(float* arr)
{
	delete arr;
}