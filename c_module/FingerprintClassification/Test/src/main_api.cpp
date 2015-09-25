#include "ApiRF.h"

int main(void){
	
	TrainPaths tPaths;
	tPaths.labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData3.csv";
	tPaths.dataPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API/RawData.csv";
	
	PredictPaths pPaths;
	pPaths.labelsPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/CSVs/RandomizedData3.csv";
	pPaths.dataPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API/RawData.csv";
	
	const char* resultPath = "D:/GoogleDrive/Projectes/GEYCE/FP/Data/out/Test_Nova_API/results.csv";

	NormalizeFitAndPredict(tPaths,pPaths,resultPath);
	
}

