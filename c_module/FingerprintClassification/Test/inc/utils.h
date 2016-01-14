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
};

#endif /* UTILS_H */