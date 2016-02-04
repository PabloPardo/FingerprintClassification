#include "LearningRF.h"
#include "AdaBoost.h"
#include "ImgTools.h"

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

struct Config
{
	bool verbose;
	ImgProcessingProperties* extractionProperties;
	AdaBoostProperties* adaBoostProperties;
	RFProperties* randomForestProperties;
};

class MainFP
{
	AdaBoost* fase1;
	LearningRF* fase2;
	ImgTools* imgTools;
public:
	MainFP();
	~MainFP();
	void InitConfig(Config cfg);
	void ExtractFingerPrint(int*, float**, unsigned char*, int, int, float*);
	void Extraction(LoadCsvParams,  const char*);
	void Normalize(const char*, const char*, const char*);
	void Fit1(LoadCsvParams, const char*, const char*);
	void Fit2(LoadCsvParams, const char*, const char*);
	void Predict1(bool*, Handle*, float*);
	void Predict2(int*, float**, Handle*, float*);
	void InitModel(Handle**, const char*);
	void ReleaseModel(Handle*);
	void ReleaseFloatPointer(float*);
	void PredictTest(LoadCsvParams, const char*, const char*, const char*);
};

