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

class MainRF
{
	LearningRF* learner;
public:
	MainRF();
	~MainRF();
	void ExtractFingerPrint(float**, unsigned char*, int w, int h, float*);
	void Extraction(const char*, const char*, const char*);
	void Fit(TrainPaths,const char*);
	void Predict(float**, PredictPaths, float*);
	void PredictTest(PredictPaths, const char*);
};

