#pragma once
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;

enum Inequality
{
	LESS_THAN,
	BIGGER_THAN
};

istream& operator>>(istream& str, Inequality& v);

struct Stump
{
	int dim;
	float thresh;
	Inequality ineq;
	float alpha;

	Stump() {}
	Stump(string& in) {};
	Stump(int& dim, float& thresh, Inequality& ineq, float& alpha)
		: dim(dim), thresh(thresh), ineq(ineq), alpha(alpha)
	{
	}
};

istream& operator>>(istream& is, Stump& en);
ostream& operator<<(ostream& os, const Stump& en);

struct AdaBoostProperties
{
	int nIterations;
	int stepsPerIteration;

	float thresh;
	Inequality ineq;

	AdaBoostProperties() {
		nIterations = 40;
		stepsPerIteration = 10;
		thresh = 1.28f;
		ineq = LESS_THAN;
	};
	friend std::ostream& operator<<(std::ostream& os, const AdaBoostProperties& prop);
};

using namespace cv;

class AdaBoost
{
	void StumpClassify(Mat*, const Mat, int, float, Inequality);
	void BuildStump(Stump*, double*, Mat*, const Mat, const Mat, const Mat);
public:
	AdaBoostProperties* prop;
	AdaBoost();
	~AdaBoost();
	void AdaboostTrainDS(vector<Stump>*, Mat*, const Mat, const Mat);
	void AdaboostTestDS(Mat*, Mat*, const Mat, vector<Stump>*);
	static void WriteToFile(const vector<Stump>*, const char*);
	static void ReadFromFile(vector<Stump>*, const char*);
};

