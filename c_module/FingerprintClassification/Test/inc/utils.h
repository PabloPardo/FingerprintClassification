#include "opencv2\core\core.hpp"
#ifndef UTILS_H
#define UTILS_H

using namespace std;
using namespace cv;

struct CsvData
{
	vector<string> headers;
	vector<string> file_names;
	Mat body;
};

class Utils
{
public:
	static bool verbose;
	static void memcpy(const char *orig, char *dest, long size);
	static long strlen(char *str);
	static char* strclone(char *str);
	static char* strconcat(char *str1, char *str2);
	static void int2str(int val, int minLen, char *ret);
	static char* dbl2str(double val, int minLen);
	static void dbl2str(double val, int minLen, char *ret);
	static double str2dbl(char *string);
	static int countLines(const char* path, char separator);
	static int loadCSV(CsvData*, const char*, char, int, int, int, bool = true);
	static void getFileNameFromPath(string* output, const string path)
	{
		int index = path.find_last_of("/") + 1;
		*output = path.substr(index);
	}
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
		throw new exception("This method only works with uniform data file");
	}
};

#endif /* UTILS_H */