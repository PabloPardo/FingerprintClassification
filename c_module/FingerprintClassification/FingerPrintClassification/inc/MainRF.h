#include "LearningRF.h"

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
	void* norMat;
	void* rTrees;
};

class MainRF
{
	LearningRF* learner;
public:
	MainRF();
	~MainRF();
	void ExtractFingerPrint(int*, float**, unsigned char*, int w, int h, float*);
	void Extraction(const char*, const char*, const char*);
	void Normalize(const char*, const char*, const char*);
	void Fit(const TrainPaths,const char*);
	void Predict(int*,float**, Handle*, float*);
	void PredictTest(PredictPaths, const char*);
	void InitModel(Handle**, const char*);
	void ReleaseModel(Handle*);
	void ReleaseFloatPointer(float*);
};

