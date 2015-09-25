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
};

class MainRF
{
	LearningRF* learner;
public:
	MainRF();
	~MainRF();
	void Extraction(const char*, const char*, const char*);
	void NormalizeFitAndPredict(TrainPaths,PredictPaths, const char*);
};

