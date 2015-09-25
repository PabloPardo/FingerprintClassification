#include "LearningRF.h"

class MainRF
{
	LearningRF* learner;
public:
	MainRF();
	~MainRF();
	void ExtractNormalizeAndFit(const char*,const char*,const char*);
	void NormalizeAndFit();
	void Fit();
	void ExtractNormalizeAndPredict();
	void NormalizeAndPredict();
	void Predict();
};

