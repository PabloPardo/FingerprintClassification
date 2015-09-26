#include "ApiRF.h"

int main(void){
	/*
	TrainPaths tPaths;
	tPaths.labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData5824-600.csv";
	tPaths.dataPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API2/RawDataTrain.csv";
	
	PredictPaths pPaths;
	pPaths.labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedDataPredict.csv";
	pPaths.dataPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API2/RawDataPredict.csv";
	
	const char* resultPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API2/results.csv";
	*/

	/*EXTRACTION**************************************************************************************/
	/* TRAIN*/
	const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData5824-600.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API3/RawDataTrain.csv";
	
	/* PREDICT*/
	/*const char* labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedDataPredict.csv";
	const char* imagesPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/Training/";
	const char* outPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API2/RawDataPredict.csv";*/
	

	Extraction(labelsPath,imagesPath,outPath);
	//NormalizeFitAndPredict(tPaths,pPaths,resultPath);
	
}

