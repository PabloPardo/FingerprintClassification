#include "utils.h"
#include <time.h>
#include <utils.h>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

using namespace cv;
//** *****************************************************************
//** ** STRING MANIPULATION METHODS
//** *****************************************************************

bool Utils::verbose;

void Utils::memcpy(const char *orig, char *dest, long size){
    for(long i=0;i<size;i++){
        dest[i] = orig[i];
    }
}

long Utils::strlen(char *str){
    long ret = 0;
    while(str[ret] != 0) ret++;
    return ret;
}

char* Utils::strconcat(char *str1,char *str2){
    long len1 = strlen(str1);
    long len2 = strlen(str2);
    long len = len1 + len2 + 1;
    
    char *ret = new char[len];
    for(long i=0;i<len1;i++) ret[i] = str1[i];
    for(long i=0;i<len2;i++) ret[len1+i] = str2[i];
    ret[len-1] = 0;
    
    return ret;
}

char* Utils::strclone(char *str){
    long len = strlen(str) + 1;
    char *ret = new char[len];
    
    for(long i=0;i<len;i++) ret[i] = str[i];
    return ret;
}

void Utils::int2str(int val, int minLen, char *ret){
	double tval = ((double)val)+0.01;

	// Fix number of significative digits
	for(int i=1;i<minLen;i++){
		tval /= 10;
	}

	// Print digits
	int pos = 0;
	for(int i=0;i<minLen;i++){
		ret[pos] = '0' + ((int)tval);
		tval = (tval - ((int)tval)) * 10;
		pos++;
	}

	ret[pos] = '\0';
}

char* Utils::dbl2str(double val, int minLen){
	char *ret = new char[10];
	dbl2str(val, minLen, ret);
	return ret;
}

void Utils::dbl2str(double val, int minLen, char *ret){
	val += 0.0000001;
	int pos = 0;

	int nint = 0;
	while(val >= 1){
		nint++;
		val /= 10;
	}

	if(nint == 0){
		ret[pos] = '0';
		pos++;
	}else{
		while(pos < nint){
			val = (val - ((int)val)) * 10;
			ret[pos] = '0' + (int)val;
			pos++;
		}
	}

	if(val > 0){
		ret[pos] = '_';
		pos++;
		while(val > 0 && pos < minLen){
			val = (val - ((int)val)) * 10;
			ret[pos] = '0' + (int)val;
			pos++;
		}
	}

	while(pos < minLen){
		ret[pos] = '0';
		pos++;
	}

	ret[pos] = '\0';
}

double Utils::str2dbl(char *string){
	double value = 0;

	bool is_decimal = false;
	double dec_factor = 1;

	char *chr = string;
	while(chr[0] != '\0'){
		if(chr[0] == '.'){
			is_decimal = true;
		}else if(is_decimal){
			dec_factor *= 10;
			value = value + ((double)(chr[0] - '0'))/dec_factor;
		}else{
			value = value*10 + (chr[0] - '0');
		}

		chr = &chr[1];
	}

	return value;
}

int Utils::countLines(const char* path, char separator) {
	int number_of_lines = 0;
	string line;
	ifstream myfile(path);

	if (!myfile.is_open())
	{
		return -1;
	}
	while (getline(myfile, line))
	{

		++number_of_lines;
	}
	myfile.close();
	return number_of_lines;
}

int Utils::loadCSV(CsvData* out, const char* csvPath, char separator, int col_begin, int col_end, bool with_headers)
{
	CsvData tmp;
	int nLines;
	
	if (verbose)
		cout << "Counting file lines... " << endl;
	nLines = countLines(csvPath, separator);
	if (verbose)
		cout << "Counting file lines completed: " << nLines << endl;
	
	ifstream ifs(csvPath, ifstream::in);
	
	if (!ifs.is_open()) 
		return -1;
	int cont;
	string line;
	string value;
	istringstream iss;
	if (with_headers)
	{
		tmp.body = Mat(nLines - 1, col_end - col_begin, CV_32F);
		getline(ifs, line);
		iss = istringstream(line);
	
		cont = 0;
	
		while (cont < col_begin)
		{
			getline(iss, value, separator);
			cont++;
		}

		for (int j = col_begin; j < col_end; j++)
		{
			getline(iss, value, separator);
			tmp.headers.push_back(value);
		}
	}
	else
		tmp.body = Mat(nLines, col_end - col_begin, CV_32F);
	
	for (int i = 0; i < tmp.body.rows; i++)
	{
		getline(ifs, line);
		iss = istringstream(line);
		cont = 0;
		while (cont < col_begin)
		{
			getline(iss, value, separator);
			cont++;
		}
		// Read Line
		for (int j = 0; j < tmp.body.cols; j++)
		{
			getline(iss, value, separator);
			tmp.body.at<float>(i, j) = (float)atof(value.c_str());
			if (strcmp(value.c_str(), "true") == 0)
				tmp.body.at<float>(i, j) = 1.;
			if (strcmp(value.c_str(),"false") == 0)
				tmp.body.at<float>(i, j) = -1.;
		}
		if (verbose && tmp.body.rows > 10 && i % (tmp.body.rows / 10) == 0)
			cout << (i > 0 ? i*100 / (tmp.body.rows) : 0) << "%" << endl;
	}
	*out = tmp;
	return 0;
}


