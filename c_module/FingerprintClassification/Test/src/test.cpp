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
void getFileNameFromPath(string* output, const string path);

string test_name = "VS2012";

void ExtractForFitAll(int argc, char* argv[])
{
	/*************************** Phase 1************************************************/
	string baseImgPathFase1 = "D:/Data/Empremptes/PNGs/2014/All/";
	string csvFileFase1 = "D:/Data/Empremptes/CSVs/Fase1/VFS/2014.csv";
	string extractedFase1 = "D:/Data/Empremptes/Extracted/Fase 1/2014-";
	
	/*************************** Phase 2************************************************/
	string baseImgPathFase2 = "D:/Data/Empremptes/PNGs (Mala Qualitat)/Train/FullSize/";
	string csvFileFase2 = "D:/Data/Empremptes/CSVs/Fase 2/RandomizedData.csv";
	string extractedFase2 = "D:/Data/Empremptes/Extracted/Fase 2/RandomizedData-";
	/*********************************************************************************/

	extractedFase1 = extractedFase1 + test_name + ".csv";
	extractedFase2 = extractedFase2 + test_name + ".csv";

	Config cfg;
	cfg.verbose = true;
	ImgProcessingProperties* extractionProperties = new ImgProcessingProperties();
	extractionProperties->n_bins = 32;
	extractionProperties->rad_grad = 1;
	extractionProperties->rad_dens = 3;
	extractionProperties->rad_entr = 5;
	cfg.extractionProperties = extractionProperties;
	InitConfig(cfg);

	LoadCsvParams paramsFase1;
	paramsFase1.baseImgPath = baseImgPathFase1.c_str();
	paramsFase1.csvFile = csvFileFase1.c_str();
	paramsFase1.separator = ';';
	paramsFase1.globalRange.begin = 3;
	paramsFase1.globalRange.end = 19;
	paramsFase1.XRange = new MatRange();
	paramsFase1.XRange->begin = 0;
	paramsFase1.XRange->end = 13;
	/*In this step, we don't need y, we only extract data from images and concatenate it with X known features to make a super matrix file */
	paramsFase1.yRange = NULL;
	paramsFase1.withHeaders = true;
	paramsFase1.fileNameIndex = 2;
	paramsFase1.folderNameIndex = 1;
	paramsFase1.fingerFieldName = "dedo";

	LoadCsvParams paramsFase2;
	paramsFase2.baseImgPath = baseImgPathFase2.c_str();
	paramsFase2.csvFile = csvFileFase2.c_str();
	paramsFase2.separator = ';';
	paramsFase2.globalRange.begin = 4;
	paramsFase2.globalRange.end = 23;
	paramsFase2.XRange = new MatRange();
	paramsFase2.XRange->begin = 6;
	paramsFase2.XRange->end = 19;
	/*In this step, we don't need y, we only extract data from images and concatenate it with X known features to make a super matrix file */
	paramsFase2.yRange = NULL;
	paramsFase2.withHeaders = true;
	paramsFase2.fileNameIndex = 2;
	paramsFase2.fingerFieldName = "dedo";

	Extraction(paramsFase1, extractedFase1.c_str());
	Extraction(paramsFase2, extractedFase2.c_str());
}

void FitAll(int argc, char* argv[])
{
	/******************************Fase 1 *****************************/
	string csvFileFase1 = "D:/Data/Empremptes/CSVs/Fase1/VFS/2014.csv";
	string extractedFileFase1 = "D:/Data/Empremptes/Extracted/Fase 1/2014-";
	string modelFase1 = "D:/Data/Empremptes/Models/Fase 1/";
	/******************************Fase 2 *****************************/
	string csvFileFase2 = "D:/Data/Empremptes/CSVs/Fase2/VFS/RandomizedData.csv";
	string extractedFileFase2 = "D:/Data/Empremptes/Extracted/Fase 2/RandomizedData-";
	string modelFase2 = "D:/Data/Empremptes/Models/Fase 2/";
	/******************************************************************/
	extractedFileFase1 = extractedFileFase1 + test_name + ".csv";
	extractedFileFase2 = extractedFileFase2 + test_name + ".csv";
	modelFase1 = modelFase1 + test_name + "/";
	modelFase2 = modelFase2 + test_name + "/";

	Config cfg;
	cfg.verbose = true;
	AdaBoostProperties* abProp = new AdaBoostProperties();
	abProp->thresh = 1.28f;
	abProp->ineq = LESS_THAN;
	abProp->nIterations = 60;
	abProp->stepsPerIteration = 10;
	cfg.adaBoostProperties = abProp;
	RFProperties* rfProp = new RFProperties();
	rfProp->max_depth = 16;
	rfProp->min_samples_count = 2;
	rfProp->max_categories = 3;
	rfProp->max_num_of_trees_in_forest = 10;
	rfProp->nactive_vars = 0;
	cfg.randomForestProperties = rfProp;
	InitConfig(cfg);

	LoadCsvParams paramsFase1;
	paramsFase1.csvFile = csvFileFase1.c_str();
	paramsFase1.separator = ';';
	paramsFase1.globalRange.begin = 3;
	paramsFase1.globalRange.end = 19;
	/* We didn't finish phase1. We just use NBIS information to train. There is no extracted data from the fingerprint image. That's why we use X range and y range */
	/* When phase1 finish, we will use extracted data concatenated with actual X and won't need XRange anymore */
	paramsFase1.XRange = NULL;
	//paramsFase1.XRange->begin = 0;
	//paramsFase1.XRange->end = 13;
	paramsFase1.yRange = new MatRange();
	paramsFase1.yRange->begin = 15;
	paramsFase1.yRange->end = 15;
	paramsFase1.withHeaders = true;
	paramsFase1.fileNameIndex = 3;
	paramsFase1.fingerFieldName = "dedo";

	LoadCsvParams paramsFase2;
	paramsFase2.csvFile = csvFileFase2.c_str();
	paramsFase2.separator = ';';
	paramsFase2.globalRange.begin = 4;
	paramsFase2.globalRange.end = 10;
	paramsFase2.XRange = NULL;
	/* We only need y in this step because we want to fit extracted data vs expected output */
	paramsFase2.yRange = new MatRange();
	paramsFase2.yRange->begin = 0;
	paramsFase2.yRange->end = 6;
	paramsFase2.withHeaders = true;

	Fit1(paramsFase1, extractedFileFase1.c_str(), modelFase1.c_str());
	//Fit2(paramsFase2, extractedFileFase2.c_str(), modelFase2.c_str());
}


void Predict_From_Extracted(int argc, char* argv[])
{
	bool paramsFromCMD = false;

	char* labelsDir = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/Malos_15_07_08.csv";
	char* inputDir = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Extracted/Malos_15_07_08-VS2012.csv";
	char* modelDir = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_VS2012_library/";
	char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_VS2012_library/results.txt";

	if (argc == 3 && paramsFromCMD)
	{
		inputDir = argv[1];
		modelDir = argv[2];
	}
	else
	{
		cout << "Usage: Test.exe <inputDir> <modelDir>" << endl;
		cout << "Loading default params..." << endl;
	}

	CsvData X_test;
	Utils::loadCSV(&X_test, inputDir, ',', 1, 924, 0, false);
	CsvData y_test;
	Utils::loadCSV(&y_test, labelsDir, ';', 1, 7, 0);

	Handle* hnd;
	ReturnType ret = InitModel(&hnd, modelDir);
	if (ret.code > 0)
	{
		cout << ret.message << endl;
		throw new exception(ret.message);
	}

	ofstream file;
	file.open(outPath);

	for (int i = 0; i < X_test.body.rows; i++)
	{
		float *probs;
		int len;
		Predict2(&len, &probs, hnd, (float*)X_test.body.row(i).data);
		string fName;
		Utils::getFileNameFromPath(&fName, y_test.file_names[i]);
		file << fName;
		for (int j = 0; j < len; j++)
			file << ";" << probs[j];
		for (int j = 0; j < len; j++)
			file << ";" << y_test.body.at<float>(i, j);
		file << endl;
	}
	file.close();
	ReleaseModel(hnd);
}

/*
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
*/
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
				ret = Predict1(&isGood, hnd, feat);
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



void RFTrain(int argc, char* argv[])
{
	LoadCsvParams params;
	params.csvFile = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData.csv";
	params.separator = ';';
	params.globalRange.begin = 4;
	params.globalRange.end = 10;
	params.XRange = NULL;
	params.yRange = new MatRange();
	params.yRange->begin = 0;
	params.yRange->end = 6;
	params.withHeaders = true;

	const char* dataPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Extracted/RandomizedData-VS2012.csv";
	const char* modelPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_VS12_Library/";
	Fit2(params, dataPath, modelPath);
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
		Predict1(&adaGood, hnd, (float*)output.row(i).data);
		file << file_name << ";" << (adaGood ? "True" : "False") << ";" << (kit4Good ? "true" : "false") << endl;
	}
	ReleaseModel(hnd);
}

int main(int argc, char* argv[]) {
	//ExtractForFitAll(argc, argv);
	FitAll(argc, argv);
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

void getFileNameFromPath(string* output, const string path)
{
	size_t i = path.rfind(path, 1);

	if (i != string::npos) {
		*output = path.substr(i + 1, path.length() - i);
	}
	else {
		*output = "";
	}
}
