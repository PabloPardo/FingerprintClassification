#include <string>

#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

#ifndef UTILS_H
#define UTILS_H

using namespace cv;
using namespace std;

struct CsvData
{
	vector<string> headers;
	vector<string> fileNames;
	Mat body;
};

struct MatRange
{
	int begin;
	int end;
};

struct LoadCsvParams
{
	bool withHeaders = false;
	const char* csvFile = 0;
	const char* baseImgPath = 0;
	char separator = 0;
	MatRange globalRange;
	MatRange* XRange;
	MatRange* yRange;
	int fileNameIndex = -1;
	int folderNameIndex = -1;
	const char* fingerFieldName = 0;;
};

//Deprecated
//enum CSV_HEADERS 
//{
//	EmId=0,
//	EmUsuari,
//	EmNomFitxer,
//	EmNumDit,
//	EmBorrosa,
//	EmPetita,
//	EmNegre,
//	EmClara,
//	EmMotejada,
//	EmDefectuosa,
//	dedo,
//	nfiq,
//	foreground,
//	numMinucias,
//	uno,
//	dos,
//	tres,
//	cuatro,
//	cinco,
//	seis,
//	siete,
//	ocho,
//	nueve
//};
////Deprecated
//struct LabelsAndFeaturesData {
//	Mat matrix;
//	vector<string> imgFileNames;
//	vector<string> imgPaths;
//	Mat features;
//};

struct Constants
{
	static const int TOTAL_FEATURES = 923;
	static const int NUM_ROW_SEGMENTS = 3;
	static const int NUM_COL_SEGMENTS = 2;
	static const int NUM_CLASSIFIERS = 6;
	static const int NUM_FEATURES = 22;
};

class Utils
{
	static int LoadCSV(CsvData*, LoadCsvParams);
	static int DigitalizeFingers(Mat*, const Mat);
public:
	static bool verbose;
	static bool has_suffix(const string &str, const string &suffix);
	static void countLines(int*, const char*, bool = false);
	static void getFileNameFromPath(string* output, const string path);
	static void throwError(string error);
	static void LoadFitDataFromFile(Mat*, Mat*, vector<string>*, LoadCsvParams);
	static void calculateTimeLeft(long*, long, int, int);
	static void convertTime(string*, long);
	template<typename T> static int saveMatToCSV(Mat mat, const char* outFile)
	{
		string fileName = outFile;
		ofstream myfile(fileName);
		if (myfile.is_open())
		{
			for (int i = 0; i < mat.rows; i++)
			{
				//myfile << imgPaths[i];
				myfile << mat.at<T>(i, 0);
				for (int j = 1; j < mat.cols; j++)
					myfile << "," << mat.at<T>(i, j);
				myfile << "\n";
			}
			myfile.close();
		}
		else
		{
			throw new exception(("Unable to open file " + fileName).c_str());
		}
	}
	template<typename T> static int loadMatFromCSV(Mat* mat, const char* inFile)
	{
		string fileName = inFile;
		ofstream myfile(fileName);
		if (myfile.is_open())
		{
			for (int i = 0; i < mat->rows; i++)
			{
				//myfile << imgPaths[i];
				myfile << mat->at<T>(i, 0);
				for (int j = 1; j < mat->cols; j++)
					myfile << "," << mat->at<T>(i, j);
				myfile << "\n";
			}
			myfile.close();
		}
		else
		{
			throw new exception(("Unable to open file " + fileName).c_str());
		}
	}

};
#endif /* UTILS_H */