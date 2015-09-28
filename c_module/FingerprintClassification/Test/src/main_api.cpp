#include "ApiRF.h"
#include <sstream>

using namespace std;
int main(void){
	
	const char* pathBase = "//ssd2015/";
	//const char* pathBase = "D:/GoogleDrive/Projectes/GEYCE/FP/";

	TrainPaths tPaths;
	tPaths.labelsPath = ((string)pathBase + "Data/CSVs/RandomizedData5824-600.csv").c_str();
	tPaths.dataPath = ((string)pathBase + "Data/out/Test_Nova_API3/RawDataTrain.csv").c_str();
	
	PredictPaths pPaths;
	pPaths.labelsPath = ((string)pathBase + "Data/CSVs/RandomizedDataPredict.csv").c_str();
	pPaths.dataPath = ((string)pathBase + "Data/out/Test_Nova_API3/RawDataPredict.csv").c_str();
	pPaths.modelDir = ((string)pathBase + "Data/out/Test_Nova_API3/").c_str();

	const char* resultPath = ((string)pathBase + "Data/out/Test_Nova_API3/results-fit-predict.csv").c_str();
	

	/*EXTRACTION**************************************************************************************/
	/* TRAIN*/
	/*const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData5824-600.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/RawDataTrain.csv";*/
	
	/* PREDICT*/
	/*const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedDataPredict.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/RawDataPredict.csv";*/
	

	//Extraction(labelsPath,imagesPath,outPath);
	Fit(tPaths,pPaths.modelDir);
	PredictTest(pPaths,resultPath);
	//NormalizeFitAndPredict(tPaths,pPaths,resultPath);
	
}

