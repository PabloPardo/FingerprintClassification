#pragma once
#include "opencv2\core\core.hpp"

using namespace cv;

enum Inequality
{
	BIGGER_THAN,
	LESS_THAN
};

struct Stump
{
	int dim;
	float thresh;
	Inequality ineq;
	float alpha;
};

class AdaBoost
{
public:
	AdaBoost();
	~AdaBoost();
	void StumpClassify(Mat*, const Mat, int, float, Inequality);
	void BuildStump(Stump*, double*, Mat*, const Mat, const Mat, const Mat, float = 10.0);
	void AdaboostTrainDS(vector<Stump>*, Mat*, const Mat, const Mat, int = 40, int = 10);
	void AdaboostTestDS(Mat*, Mat*, const Mat, const vector<Stump>, int = 0, Inequality = LESS_THAN);
};

