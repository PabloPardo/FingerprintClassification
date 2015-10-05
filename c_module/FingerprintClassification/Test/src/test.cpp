#include <windows.h>
#include "apirf.h"
#include <iostream>
#include <string>
#include "FoldSplitter.h"
#include "utils.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\operations.hpp"
#include "opencv2\ml\ml.hpp"

using namespace std;
using namespace cv;

LList *getDirFiles(char *dir);
void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos);

void Extract_Main(int argc, char* argv[])
{
	if (argc != 4)
		cout << "Usage: Test.exe <labelsPath> <imagesDir> <outputFile>" << endl;

	Extraction(argv[1], argv[2], argv[3]);
}

void Fit_And_Predict_Main()
{
	string pathBase = "D:/GoogleDrive/Projectes/GEYCE/FP/";

	TrainPaths tPaths;
	string* res = new string(pathBase + "Data/CSVs/RandomizedData.csv");
	tPaths.labelsPath = res->c_str();
	res = new string(pathBase + "Data/out/Test_Nova_API5/RawDataTrain.csv");
	tPaths.dataPath = res->c_str();

	PredictPaths pPaths;
	res = new string(pathBase + "Data/CSVs/Malos_15_07_08.csv");
	pPaths.labelsPath = res->c_str();
	res = new string(pathBase + "Data/out/Test_Nova_API5/RawDataPredict.csv");
	pPaths.dataPath = res->c_str();
	res = new string(pathBase + "Data/out/Test_Nova_API5/");
	pPaths.modelDir = res->c_str();

	res = new string(pathBase + "Data/out/Test_Nova_API5/results-fit-predict.csv");
	const char* resultPath = res->c_str();

	Fit(tPaths, pPaths.modelDir);
	PredictTest(pPaths, resultPath);
}

void Predict_Main(int argc, char* argv[])
{
	if (argc != 3)
		cout << "Usage: Test.exe <modelDir> <predictImagesDir>" << endl;

	Handle* hnd;
	ReturnType ret = InitModel(&hnd, argv[1]);
	if (ret.code > 0)
	{
		cout << ret.message << endl;
		throw new exception(ret.message);
	}

	LList *files = getDirFiles((char *)argv[2]);
	if (files != NULL)
	{
		LList *fp = files;
		while (false)
		{
			try
			{
				FName data = fp->element;
				Mat in = imread((char *)data.fpath, cv::IMREAD_GRAYSCALE);
				int len;
				float* features;
				float dummy_nfiqs[] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
				ret = ExtractFingerPrint(&len, &features, in.data, in.rows, in.cols, dummy_nfiqs);
				if (ret.code > 0)
					throw new exception(ret.message);
				float* probs;
				ret = Predict(&len, &probs, hnd, features);
				if (ret.code > 0)
					throw new exception(ret.message);
				ret = ReleaseFloatArrayPointer(features);
				if (ret.code > 0)
					throw new exception(ret.message);
				ret = ReleaseFloatArrayPointer(probs);
				if (ret.code > 0)
					throw new exception(ret.message);
			}
			catch (Exception cvEx)
			{
				cout << cvEx.what() << endl;
			}
			catch (exception stdEx)
			{
				cout << stdEx.what() << endl;
			}
			catch (...)
			{
				cout << "Unknown error" << endl;
			}
		}
		ret = ReleaseModel(hnd);

	}



}

int main(int argc, char* argv[]){

	/*for (int i = 0; i < 1000; i++)
	{ 
		CvRTrees* arr = new CvRTrees[6];
		delete[] arr;
	}
	system("pause");*/

	//Handle* hnd;

	//InitModel(&hnd,argv[1]);
	//ReleaseModel(hnd);

	Predict_Main(argc,argv);

	//Fit_And_Predict_Main();

	//string pathBase = "//ssd2015/DataFase1/Empremptes/";
	//const char* pathBase = "D:/GoogleDrive/Projectes/GEYCE/FP/";

	/*TrainPaths tPaths;
	tPaths.labelsPath = ((string)pathBase + "Data/CSVs/RandomizedData5824-600.csv").c_str();
	tPaths.dataPath = ((string)pathBase + "Data/out/Test_Nova_API3/RawDataTrain.csv").c_str();

	PredictPaths pPaths;
	pPaths.labelsPath = ((string)pathBase + "Data/CSVs/RandomizedDataPredict.csv").c_str();
	pPaths.dataPath = ((string)pathBase + "Data/out/Test_Nova_API3/RawDataPredict.csv").c_str();
	pPaths.modelDir = ((string)pathBase + "Data/out/Test_Nova_API3/").c_str();

	const char* resultPath = ((string)pathBase + "Data/out/Test_Nova_API3/results-fit-predict.csv").c_str();*/


	/*EXTRACTION**************************************************************************************/
	/* TRAIN*/
	/*const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData5824-600.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/RawDataTrain.csv";*/

	/* PREDICT*/
	/*const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedDataPredict.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/RawDataPredict.csv";*/

	/**********************************************/
	/* Fase 1*/
	/*********************************************/

	//string labelsPath = (pathBase + "CSVs/Errores.csv").c_str();
	//string labelsPath = (pathBase + "CSVs/AdaBoost15.csv").c_str();
	//string imagesPath = (pathBase + "PNGs/2014/All/").c_str();
	//string outPath = (pathBase + "Extracted/Errores.csv").c_str();

	//Extraction(labelsPath.c_str(),imagesPath.c_str(),outPath.c_str());
	//Fit(tPaths,pPaths.modelDir);
	//PredictTest(pPaths,resultPath);
	//NormalizeFitAndPredict(tPaths,pPaths,resultPath);

}

//** *****************************************************************
//** ** HELPER METHODS
//** *****************************************************************

/* getDirFiles(dir)
*
* Lists all of the files inside of the path specified by 'dir' and returns them
* inside of a linked list structure, where each element of the linked list is a
* structure containing the name and path of the file.
*
*     dir:    Pointer to a character array specifying the path where the files
*             are to be found.
*/
LList *getDirFiles(char *dir){
	LList *ret = NULL, *cpos = NULL;
	char *sst;

	// List PNG files
	sst = strconcat(dir, "*.png");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List JPG files
	sst = strconcat(dir, "*.jpg");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List BIN files
	sst = strconcat(dir, "*.bin");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	return ret;
}

void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos){
	WIN32_FIND_DATA ffd;

	// Get handle to first file
	HANDLE hFind = FindFirstFile((char *)pattern, &ffd);
	if (hFind == INVALID_HANDLE_VALUE) return;

	// Build linked list with file paths
	BOOL hasNext = true;
	while (hasNext){
		if (*ret == NULL){
			*ret = new LList;
			*cpos = *ret;
		}
		else{
			(*cpos)->next = new LList;
			*cpos = (*cpos)->next;
		}

		(*cpos)->element.fname = strclone(ffd.cFileName);
		(*cpos)->element.fpath = strconcat(dir, ffd.cFileName);
		hasNext = FindNextFile(hFind, &ffd);
	}
}