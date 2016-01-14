#include <windows.h>
#include "apiF1F2.h"
#include <iostream>
#include <fstream>
#include <string>
#include "FoldSplitter.h"
#include "utils.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\operations.hpp"
#include "opencv2\ml\ml.hpp"
#include <time.h>

using namespace std;
using namespace cv;

LList *getDirFiles(char *dir);
void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos);
int digitalizeFingers(Mat* output, const Mat onlyFingerNumber);

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

	Fit2(tPaths, pPaths.modelDir);
	PredictTest(pPaths, resultPath);
}

void Predict_Main(int argc, char* argv[])
{
	bool paramsFromCMD = false;
	char* modelDir = "//ssd2015/Data/out/Test_Nova_API5/";
	char* predictImagesDir = "//ssd2015/Data/PredictData/";

	if (argc == 3 && paramsFromCMD)
	{
		modelDir = argv[1];
		predictImagesDir = argv[2];
	}
	else
	{
		cout << "Usage: Test.exe <modelDir> <predictImagesDir>" << endl;
		cout << "Loading default params..." << endl;
	}
	
	Handle* hnd;
	ReturnType ret = InitModel(&hnd, modelDir);
	if (ret.code > 0)
	{
		cout << ret.message << endl;
		throw new exception(ret.message);
	}

	LList *files = getDirFiles(predictImagesDir);
	clock_t begin, end;
	if (files != NULL)
	{
		LList *fp = files;
		while (fp != NULL)
		{
			try
			{
				begin = clock();
				FName data = fp->element;
				Mat in = imread((char *)data.fpath, cv::IMREAD_GRAYSCALE);
				
				cout << "Processant imatge:" << data.fpath << " en ";
				int len;
				float* features;
				float dummy_nfiqs[] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
				ret = ExtractFingerPrint(&len, &features, in.data, in.rows, in.cols, dummy_nfiqs);
				if (ret.code > 0)
					throw new exception(ret.message);
				float* probs;
				bool isGood;
				float feat[22];
				ret = Predict1(&isGood, hnd, 1.28f, feat);
				ret = Predict2(&len, &probs, hnd, features);
				if (ret.code > 0)
					throw new exception(ret.message);
				ret = ReleaseFloatArrayPointer(features);
				if (ret.code > 0)
					throw new exception(ret.message);
				ret = ReleaseFloatArrayPointer(probs);
				if (ret.code > 0)
					throw new exception(ret.message);
				end = clock();
				cout << double(end - begin) / CLOCKS_PER_SEC << " sec" << endl;
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
			fp = fp->next;
		}
		ret = ReleaseModel(hnd);
		system("pause");
	}
}

/*****************************Fase I*****************************/
void ABTrain(int argc, char* argv[])
{
	LoadCsvParams params;
	params.csvFile = argv[1];
	params.separator = argv[2][0];
	params.begin_header = atoi(argv[3]);
	params.end_header = atoi(argv[4]);
	params.with_headers = atoi(argv[5]);
	Fit1(params, "./", 60, 10);
}

void ABTrain2(int argc, char* argv[])
{
	/*ProcessedCSV* params = new ProcessedCSV();
	params->X.csvFile = "X_out_train.csv";
	params->X.separator = ';';
	params->X.begin_header = 1;
	params->X.end_header = 22;
	params->X.with_headers = false;

	params->y.csvFile = "y_out_train.csv";
	params->y.separator = ';';
	params->y.begin_header = 1;
	params->y.end_header = 2;
	params->y.with_headers = false;
	
	Fit1(params, "./", 40, 10);*/
}

void ABTest(int argc, char* argv[])
{
	bool paramsFromCMD = false;
	char* pathCsv = "//ssd2015/DataFase1/Empremptes/CSVs/Fase1/VFS/201503.csv";
	char separator = ';';
	int col_ini = 3;
	int col_end = 19;
	float thresh = 1.28f;
	
	
	if (argc > 1 && paramsFromCMD)
	{
		pathCsv = argv[1];
		if (argc > 2)
		{
			separator = argv[2][0];
			if (argc > 3)
			{
				col_ini = atoi(argv[3]);
				if (argc > 4)
				{
					col_end = atoi(argv[4]);
					if (argc > 5)
					{
						thresh = atof(argv[5]);
					}
				}
			}
		}
	}
	Utils::verbose = true;
	int ret = 0;
	clock_t begin, end;

	CsvData data;

	cout << "Load CSV " << pathCsv << " ..." << endl;
	begin = clock();
	vector<string> file_names;
	ret = Utils::loadCSV(&data, pathCsv, separator, col_ini, col_end, 3);
	
	end = clock();
	cout << double(end - begin) / CLOCKS_PER_SEC << "sec" << endl;

	Mat geyce_y, kit4_y;
	
	vector<string>::iterator it;
	it = find(data.headers.begin(), data.headers.end(), "GEYCEResult");
	int ind = (int)distance(data.headers.begin(), it);
	geyce_y = data.body.col(ind);

	it = find(data.headers.begin(), data.headers.end(), "SAGEMResult");
	ind = (int)distance(data.headers.begin(), it);
	kit4_y = data.body.col(ind);

	data.body = data.body.colRange(0, 13);

	it = find(data.headers.begin(), data.headers.end(), "dedo");
	int posDedo = (int)distance(data.headers.begin(), it);
	Mat digFingers;
	digitalizeFingers(&digFingers, data.body.col(posDedo));

	Mat output = data.body.colRange(0, posDedo);
	if (output.cols == 0)
		output = data.body.colRange(posDedo + 1, data.body.cols);
	else
		hconcat(output, data.body.colRange(posDedo + 1, data.body.cols), output);
	hconcat(digFingers, output, output);

	Handle* hnd;
	InitModel(&hnd, "./");
	
	ofstream file;
	file.open("results.txt");
	file << "file;obtained;expected" << endl;
	for (int i = 0; i < output.rows; i++)
	{
		bool kit4Good = kit4_y.row(i).at<float>(0, 0) > 0.0f;
		bool adaGood;
		
		/*for (int k = 0; k < output.row(i).cols; k++)
			cout << output.row(i).at<float>(0, k) << " ";*/
		string file_name = data.file_names.at(i);
		Predict1(&adaGood, hnd, thresh, (float*)output.row(i).data);
		file << file_name << ";" << (adaGood ? "True" : "False") << ";" << (kit4Good ? "true" : "false") << endl;
	}
	ReleaseModel(hnd);
}

int main(int argc, char* argv[]){
	//ABTrain(argc, argv);
	//ABTest(argc, argv);

	/*long size = 1179855 * 914;
	float* f = new float[size];*/

	/*for (int i = 0; i < 1000; i++)
	{ 
		CvRTrees* arr = new CvRTrees[6];
		delete[] arr;
	}
	system("pause");*/

	//Handle* hnd;

	//InitModel(&hnd,argv[1]);
	//ReleaseModel(hnd);

	/*const char* inputDir = "D:/Data/Empremptes/Extracted";
	const char* inputFile = "201411_0404.csv";
	const char* outputDir = "D:/Data/Empremptes/Extracted";

	Normalize(inputDir, inputFile, outputDir);*/

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
	sst = Utils::strconcat(dir, "*.png");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List JPG files
	sst = Utils::strconcat(dir, "*.jpg");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List BIN files
	sst = Utils::strconcat(dir, "*.bin");
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

		(*cpos)->element.fname = Utils::strclone(ffd.cFileName);
		(*cpos)->element.fpath = Utils::strconcat(dir, ffd.cFileName);
		hasNext = FindNextFile(hFind, &ffd);
	}
}

int digitalizeFingers(Mat* output, const Mat onlyFingerNumber)
{
	Mat tmp = Mat(onlyFingerNumber.rows, 10, onlyFingerNumber.type());
	try
	{
		for (int i = 0; i < tmp.rows; i++)
		{
			int nFinger = (int)onlyFingerNumber.at<float>(i, 0);
			for (int j = 0; j < tmp.cols; j++)
			{
				if (j + 1 != nFinger)
					tmp.at<float>(i, j) = 0;
				else
					tmp.at<float>(i, j) = 1;
			}
		}
	}
	catch (...)
	{
		return -1;
	}
	*output = tmp;
	return 0;
}