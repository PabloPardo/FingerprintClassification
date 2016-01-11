#pragma once
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;

struct CsvData
{
	vector<string> headers;
	Mat body;
};

struct LoadCsvParams
{
	bool with_headers;
	char* csvFile;
	char separator;
	int begin_header;
	int end_header;
};

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

using namespace cv;

class AdaBoost
{
public:
	static bool verbose;
	AdaBoost();
	~AdaBoost();
	void StumpClassify(Mat*, const Mat, int, float, Inequality);
	void BuildStump(Stump*, double*, Mat*, const Mat, const Mat, const Mat, int = 10);
	void AdaboostTrainDS(vector<Stump>*, Mat*, const Mat, const Mat, int = 40, int = 10);
	void AdaboostTestDS(Mat*, Mat*, const Mat, vector<Stump>*, float = 0, Inequality = LESS_THAN);
	static void WriteToFile(const vector<Stump>*, const char*);
	static void ReadFromFile(vector<Stump>*, const char*);
	static int LoadCSV(CsvData*, LoadCsvParams);
	int DigitalizeFingers(Mat*, const Mat);
};

