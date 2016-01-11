#include <vector>
#include "opencv2\core\core.hpp"

#ifndef APIDS_H
#define	APIDS_H

struct ReturnType{
	int         code;
	const char* message;
};

enum Inequality
{
	LESS_THAN,
	BIGGER_THAN
};

struct Stump
{
	int dim;
	float thresh;
	Inequality ineq;
	float alpha;
};

using namespace std;
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType SetVerbose(bool);
	__declspec(dllimport) ReturnType Train(vector<Stump>*, Mat*, const Mat, const Mat, int, int);
	__declspec(dllimport) ReturnType Test(Mat*, Mat*, const Mat, vector<Stump>*, float, Inequality = LESS_THAN);
	__declspec(dllimport) void WriteToFile(const vector<Stump>*, const char*);
	__declspec(dllimport) void ReadFromFile(vector<Stump>*, const char*);
#ifdef __cplusplus
}
#endif

#endif