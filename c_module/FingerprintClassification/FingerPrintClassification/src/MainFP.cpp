#include "MainFP.h"
#include "utils.h"
#include "LearningRF.h"
#include <time.h>
#include <fstream>
#include <io.h>
#include "opencv2\ml\ml.hpp"
#include "opencv2\core\core.hpp"

using namespace cv;

MainFP::MainFP()
{
	fase1 = new AdaBoost();
	fase1->verbose = true;
	fase2 = new LearningRF();
	fase2->prop->verbose = true;
}

MainFP::~MainFP()
{
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

void MainFP::Extraction(const char* labelsAndFeaturesPath, const char* imagesPath, const char* outPath)
{
	LabelsAndFeaturesData lfData = readCSV(labelsAndFeaturesPath, imagesPath);
	
	Mat rawFeaturesWithoutNFIQ;
	fase2->Extract(lfData.imgPaths, lfData.imgFileNames, &rawFeaturesWithoutNFIQ);

	Mat rawFeatures;

	concatMatrix(rawFeaturesWithoutNFIQ, lfData.features, &rawFeatures);
	
	exportFileFeatures(rawFeatures, lfData.imgFileNames, outPath);
}

void MainFP::ExtractFingerPrint(int* lenFeatures, float** features, unsigned char* img, int w, int h, float* nfiqFeatures)
{
	Mat in = Mat(w, h, CV_8U, img);
	Mat rawFeatures;
	fase2->ImageExtraction(in, &rawFeatures);
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
	importFileFeatures(&fNames, &trainData, dataPath.c_str(), fase2->prop->verbose, Constants::TOTAL_FEATURES);
	fase2->CreateNorm(trainData, &normalization, &normTrainData);
	exportFileFeatures(normTrainData, fNames, ((string)outputDir + "/Norm_" + dataFile).c_str());
	saveNormalization(normalization, ((string)outputDir + "/normalization.csv").c_str());
}

void MainFP::LoadFitDataFromFile(Mat* X_out, Mat* y_out, LoadCsvParams impData)
{
	/*switch (impData->csvType)
	{
	case GEYCE:
	{
		GeyceCSV* gycCSV = (GeyceCSV*)impData;*/
		CsvData csv_data;
		fase1->LoadCSV(&csv_data, impData);
		Mat kit4_y;
		vector<string>::iterator it;

		it = find(csv_data.headers.begin(), csv_data.headers.end(), "SAGEMResult");
		int ind = (int)distance(csv_data.headers.begin(), it);
		kit4_y = csv_data.body.col(ind);
		*y_out = kit4_y;

		csv_data.body = csv_data.body.colRange(0, 13);

		it = find(csv_data.headers.begin(), csv_data.headers.end(), "dedo");
		int posDedo = (int)distance(csv_data.headers.begin(), it);
		Mat digFingers;
		fase1->DigitalizeFingers(&digFingers, csv_data.body.col(posDedo));

		Mat output = csv_data.body.colRange(0, posDedo);
		if (output.cols == 0)
			output = csv_data.body.colRange(posDedo + 1, csv_data.body.cols);
		else
			hconcat(output, csv_data.body.colRange(posDedo + 1, csv_data.body.cols), output);
		hconcat(digFingers, output, output);
		*X_out = output;
	//	break;
	//}
	//case PROCESSED:
	//{
	//	ProcessedCSV* xy = (ProcessedCSV*)impData;
	//	CsvData X, y;
	//	//AdaBoost::LoadCSV(&X, params);
	//	//AdaBoost::LoadCSV(&y, xy->y);
	//	*X_out = X.body;
	//	*y_out = y.body;
	//	break;
	//}
	//}
}

void MainFP::Fit1(LoadCsvParams inputParams, const char* outputPath, int num_it, int num_ds_steps)
{
	vector<Stump> weak_class_arr;
	Mat agg_class_est;
	Mat norMat, normalization;
	Mat X, y;
	LoadFitDataFromFile(&X, &y, inputParams);
	fase2->CreateNorm(X, &normalization, &norMat);
	fase1->AdaboostTrainDS(&weak_class_arr, &agg_class_est, norMat, y, num_it, num_ds_steps);
	AdaBoost::WriteToFile(&weak_class_arr, ((string)outputPath + "/weak_class_arr.csv").c_str());
	saveNormalization(normalization, ((string)outputPath + "/norFase1.csv").c_str());
}

void MainFP::Fit2(const TrainPaths tPaths, const char* outputDir)
{
	Mat normalization;
	CvRTrees* rtrees;
	LabelsAndFeaturesData lfTData = readCSV(tPaths.labelsPath);	
	Mat trainData;
	vector<string> fNames;
	importFileFeatures(&fNames, &trainData, tPaths.dataPath,false,Constants::TOTAL_FEATURES);
				
	Mat normTrainData;
	
	fase2->CreateNorm(trainData, &normalization, &normTrainData);
	
	fase2->Fit(&rtrees, lfTData.matrix, normTrainData);
	
	for(int i = 0; i < Constants::NUM_CLASSIFIERS; i++)
	{
		char fileName[10000];
		sprintf_s(fileName, "%smodel_%d.xml", outputDir, i);
		rtrees[i].save(fileName);
	}

	saveNormalization(normalization, ((string)outputDir + "/normalization.csv").c_str());
}

void MainFP::PredictTest(PredictPaths pPaths, const char* results)
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
	loadNormalization(&norMat,((string)pPaths.modelDir + "/normalization.csv").c_str(), Constants::TOTAL_FEATURES);
	

	LabelsAndFeaturesData lfPData = readCSV(pPaths.labelsPath);	
	Mat predictData;
	importFileFeatures(NULL, &predictData, pPaths.dataPath, false, Constants::TOTAL_FEATURES);
	Mat normPredictData;
	fase2->Normalize(predictData, &normPredictData, norMat);
		
	ofstream file;
	file.open(results);

	for(int i = 0; i < normPredictData.rows; i++)
	{
		float *probs;
		Mat rowi = Mat(normPredictData.row(i));
		fase2->Predict(&probs, models, rowi);
		file << lfPData.imgFileNames[i];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << probs[j];
		for(int j = 0; j < Constants::NUM_CLASSIFIERS; j++)
			file << ";" << lfPData.matrix.at<int>(i,j);
		file << endl;
	}
	file.close();
}

void MainFP::Predict1(bool* enoughQuality, Handle* hnd, float thresh, float* features)
{
	vector<Stump>* weak_class_arr = (vector<Stump>*)hnd->vec_stumps;
	Mat* ptrMat = (Mat*)hnd->norFase1;
	Mat nM = *ptrMat;

	Mat predictData = Mat(1, Constants::NUM_FEATURES, CV_32F, features);
	Mat X;
	fase2->Normalize(predictData, &X, nM);
	Mat Y;
	Mat agg_class_est_test;
	fase1->AdaboostTestDS(&Y, &agg_class_est_test, X, weak_class_arr, thresh);
	*enoughQuality = Y.at<float>(0, 0) > 0.0f;
}

void MainFP::Predict2(int* lenProbs, float** probs, Handle* hnd, float* features)
{
	CvRTrees* models = (CvRTrees*)hnd->rTrees;

	Mat* ptrMat = (Mat*)hnd->norMat;

	Mat nM = *ptrMat;
	
	Mat predictData = Mat(1,Constants::TOTAL_FEATURES,CV_32F, features);
	Mat normPredictData;
	fase2->Normalize(predictData, &normPredictData, nM);
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
	loadNormalization(&nM, ((string)modelPath + "/normalization.csv").c_str(), Constants::TOTAL_FEATURES);
	loadNormalization(&nFase1, ((string)modelPath + "/norFase1.csv").c_str(), Constants::NUM_FEATURES);
	ret->rTrees = models;
	ret->norMat = new Mat(nM);
	ret->norFase1 = new Mat(nFase1);
	vector<Stump> ptr;
	AdaBoost::ReadFromFile(&ptr, ((string)modelPath + "/weak_class_arr.csv").c_str());
	ret->vec_stumps = new vector<Stump>(ptr);
	*hnd = ret;
	
	if (fase2->prop->verbose)
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