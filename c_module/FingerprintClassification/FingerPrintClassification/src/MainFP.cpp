#include "MainFP.h"
#include "utils.h"
#include "LearningRF.h"
#include <time.h>
#include <fstream>
#include <io.h>
#include "opencv2\ml\ml.hpp"
#include "opencv2\core\core.hpp"
#include "NorManagement.h"

using namespace cv;

MainFP::MainFP()
{
	imgTools = new ImgTools();
	fase1 = new AdaBoost();
	fase2 = new LearningRF();
}

MainFP::~MainFP()
{
	delete imgTools;
	delete fase1;
	delete fase2;
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

void MainFP::InitConfig(Config cfg)
{
	Utils::verbose = cfg.verbose;
	imgTools->prop = cfg.extractionProperties;
	fase1->prop = cfg.adaBoostProperties;
	fase2->prop = cfg.randomForestProperties;
}

void MainFP::Extraction(LoadCsvParams csvParams, const char* outPath)
{
	Mat X_out, y_out;
	vector<string> imgPaths_out;
	Utils::LoadFitDataFromFile(&X_out, &y_out, &imgPaths_out, csvParams);
	
	Mat rawFeaturesWithoutNFIQ;
	imgTools->Extract(imgPaths_out, &rawFeaturesWithoutNFIQ);

	Mat rawFeatures;

	concatMatrix(rawFeaturesWithoutNFIQ, X_out, &rawFeatures);
	
	imgTools->ExportImageFeatures(rawFeatures, imgPaths_out, outPath);
}

void MainFP::ExtractFingerPrint(int* lenFeatures, float** features, unsigned char* img, int w, int h, float* nfiqFeatures)
{
	Mat in = Mat(w, h, CV_8U, img);
	Mat rawFeatures;
	imgTools->ImageExtraction(in, &rawFeatures);
	Mat nfiq = Mat(1,Constants::NUM_FEATURES, CV_32F, nfiqFeatures);
	Mat ret;
	concatMatrix(rawFeatures,nfiq,&ret);
	
	*lenFeatures = ret.cols;
	*features = new float[*lenFeatures];
	memcpy_s(*features, (*lenFeatures)*sizeof(float), (float*)ret.data, (*lenFeatures)*sizeof(float));
}

void MainFP::Normalize(const char* inputDir, const char* dataFile, const char* outputDir)
{
	Mat normalization;
	Mat normTrainData;
	Mat trainData;
	vector<string> fNames;
	string dataPath = ((string)inputDir + "/" + (string)dataFile);
	imgTools->ImportImageFeatures(&fNames, &trainData, dataPath.c_str(), Constants::TOTAL_FEATURES);
	NorManagement::CreateNorm(trainData, &normalization, &normTrainData);
	imgTools->ExportImageFeatures(normTrainData, fNames, ((string)outputDir + "/Norm_" + dataFile).c_str());
	NorManagement::SaveNormalization(normalization, ((string)outputDir + "/normalization.csv").c_str());
}

void MainFP::Fit1(LoadCsvParams inputParams, const char* extractedFile, const char* outputPath)
{
	vector<Stump> weak_class_arr;
	Mat agg_class_est;
	Mat normalization;
	Mat X, y;
	vector<string> imgPaths;
	
	Utils::LoadFitDataFromFile(&X, &y, &imgPaths, inputParams);
	
	Mat trainData;
	vector<string> fNames;
	imgTools->ImportImageFeatures(&fNames, &trainData, extractedFile, Constants::TOTAL_FEATURES);

	Mat normTrainData;
	
	NorManagement::CreateNorm(trainData, &normalization, &normTrainData);
	
	fase1->AdaboostTrainDS(&weak_class_arr, &agg_class_est, normTrainData, y);
	AdaBoost::WriteToFile(&weak_class_arr, ((string)outputPath + "/weak_class_arr.csv").c_str());
	NorManagement::SaveNormalization(normalization, ((string)outputPath + "/norFase1.csv").c_str());
}

void MainFP::Fit2(LoadCsvParams imParams, const char* dataPath, const char* outputDir)
{
	Mat normalization;
	CvRTrees* rtrees;
	Mat X, y;
	vector<string> imgPaths;
	Utils::LoadFitDataFromFile(&X, &y, &imgPaths, imParams);

	Mat trainData;
	vector<string> fNames;
	imgTools->ImportImageFeatures(&fNames, &trainData, dataPath, Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	NorManagement::CreateNorm(trainData, &normalization, &normTrainData);
	
	fase2->Fit(&rtrees, y, normTrainData);
	
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char fileName[10000];
		sprintf_s(fileName, "%smodel_%d.xml", outputDir, i);
		rtrees[i].save(fileName);
	}

	NorManagement::SaveNormalization(normalization, ((string)outputDir + "/normalization.csv").c_str());
}

void MainFP::Predict1(bool* enoughQuality, Handle* hnd, float* features)
{
	vector<Stump>* weak_class_arr = (vector<Stump>*)hnd->vec_stumps;
	Mat* ptrMat = (Mat*)hnd->norFase1;
	Mat nM = *ptrMat;

	Mat predictData = Mat(1, Constants::NUM_FEATURES, CV_32F, features);
	Mat X;
	NorManagement::Normalize(predictData, &X, nM);
	Mat Y;
	Mat agg_class_est_test;
	fase1->AdaboostTestDS(&Y, &agg_class_est_test, X, weak_class_arr);
	*enoughQuality = Y.at<float>(0, 0) > 0.0f;
}

void MainFP::Predict2(int* lenProbs, float** probs, Handle* hnd, float* features)
{
	CvRTrees* models = (CvRTrees*)hnd->rTrees;

	Mat* ptrMat = (Mat*)hnd->norMat;

	Mat nM = *ptrMat;
	
	Mat predictData = Mat(1,Constants::TOTAL_FEATURES,CV_32F, features);
	Mat normPredictData;
	NorManagement::Normalize(predictData, &normPredictData, nM);
	fase2->Predict(probs, models, normPredictData);
	*lenProbs = Constants::NUM_CLASSIFIERS;
}

void MainFP::InitModel(Handle** hnd, const char *modelPath)
{
	clock_t begin;
	double load_rtree = 0;
	Handle* ret = new Handle();
	
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	
	for (int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
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
			//throw exception(err.c_str());
		}
	}

	Mat nM, nFase1;
	NorManagement::LoadNormalization(&nM, ((string)modelPath + "/normalization.csv").c_str(), Constants::TOTAL_FEATURES);
	NorManagement::LoadNormalization(&nFase1, ((string)modelPath + "/norFase1.csv").c_str(), Constants::NUM_FEATURES);
	ret->rTrees = models;
	ret->norMat = new Mat(nM);
	ret->norFase1 = new Mat(nFase1);
	vector<Stump> ptr;
	AdaBoost::ReadFromFile(&ptr, ((string)modelPath + "/weak_class_arr.csv").c_str());
	ret->vec_stumps = new vector<Stump>(ptr);
	*hnd = ret;
	
	if (Utils::verbose)
		cout << "Load Trees[" << load_rtree << "]" << endl;
}

void MainFP::ReleaseModel(Handle* hnd)
{	
	delete (Mat*)hnd->norMat;
	delete (Mat*)hnd->norFase1;
	delete[] (CvRTrees*)hnd->rTrees;
	delete (Handle*)hnd;
	cout << "Handle fully released..." << endl;
}

void MainFP::ReleaseFloatPointer(float* arr)
{
	delete arr;
}

void MainFP::PredictTest(LoadCsvParams params, const char* modelDir, const char* dataPath, const char* results)
{
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];

	for (int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		// Load the trained model
		char modelName[10000];
		sprintf_s(modelName, "%smodel_%i.xml", modelDir, i);
		models[i].load(modelName);
	}

	Mat norMat;
	NorManagement::LoadNormalization(&norMat, ((string)modelDir + "/normalization.csv").c_str(), Constants::TOTAL_FEATURES);

	Mat X, y;
	vector<string> imgPaths;
	Utils::LoadFitDataFromFile(&X, &y, &imgPaths, params);
	Mat predictData;
	imgTools->ImportImageFeatures(NULL, &predictData, dataPath, Constants::TOTAL_FEATURES);
	Mat normPredictData;
	NorManagement::Normalize(predictData, &normPredictData, norMat);

	ofstream file;
	file.open(results);

	for (int i = 0; i < normPredictData.rows; i++)
	{
		float *probs;
		Mat rowi = Mat(normPredictData.row(i));
		fase2->Predict(&probs, models, rowi);
		string fName;
		Utils::getFileNameFromPath(&fName, imgPaths[i]);
		file << fName;
		for (int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << probs[j];
		for (int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << y.at<float>(i, j);
		file << endl;
	}
	file.close();
}