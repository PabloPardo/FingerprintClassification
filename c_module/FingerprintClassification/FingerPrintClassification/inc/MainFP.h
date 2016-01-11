#include "LearningRF.h"
#include "AdaBoost.h"

struct TrainPaths
{
	char* labelsPath;
	char* dataPath;
};

struct PredictPaths
{
	char* labelsPath;
	char* dataPath;
	char* modelDir;
};


struct Handle
{
	/*Fase 1*/
	void* norFase1;
	void* vec_stumps;
	/*Fase 2*/
	void* norMat;	
	void* rTrees;
};

class MainFP
{
	AdaBoost* fase1;
	LearningRF* fase2;
	void LoadFitDataFromFile(Mat*, Mat*, LoadCsvParams);
public:
	MainFP();
	~MainFP();
	void ExtractFingerPrint(int*, float**, unsigned char*, int, int, float*);
	void Extraction(const char*, const char*, const char*);
	void Normalize(const char*, const char*, const char*);
	void Fit1(LoadCsvParams, const char*, int, int);
	void Fit2(const TrainPaths, const char*);
	void Predict1(bool*, Handle*, float, float*);
	void Predict2(int*, float**, Handle*, float*);
	void PredictTest(PredictPaths, const char*);
	void InitModel(Handle**, const char*);
	void ReleaseModel(Handle*);
	void ReleaseFloatPointer(float*);
};

