#include <time.h>
#include <utils.h>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\highgui\highgui.hpp"

using namespace std;
using namespace cv;


bool Utils::verbose;

void Utils::throwError(string error) {
	string err = error;
	cerr << err << endl;
	throw exception(err.c_str());
}

/**************************************************************************
*								  readCSV
*								  -------
*		path  : Path to csv file.
*		head  : Boolean to determine whether the csv file has heather or not.
*
***************************************************************************
void Utils::readCSV(LabelsAndFeaturesData* out, const char *path, const char* imagesBasePath){

	// Count the number of lines in the file to allocate memory.
	int n_lines;
	countLines(&n_lines, path);
	if(n_lines == 0)
		throwError("ERROR: CSV " + (string)path + " contains no data");
	// Use string instead of char*
	string spath(path);

	// Open CSV file
	ifstream ifs(path, ifstream::in);
	if (!ifs.is_open()) {
		throwError("ERROR: file " + spath + " could not be opened. Is the path okay?");
	}

	string line;

	// Read File
	string value;
	//string dir = spath.substr(0, spath.find_last_of("\\/"));
	
	Mat M(n_lines, Constants::NUM_CLASSIFIERS, CV_32SC1);
	
	Mat F(n_lines, Constants::NUM_FEATURES, CV_32F);
	
	vector<string> imgFileNames(0);
	vector<string> imgPaths(0);
	
	int lcnt = 0;

	int head_ruta_fitxer = 1;
	int head_nom_fitxer = 2;
	int head_img_labels_ini = 3;
	int head_img_labels_end = 9;
	int head_features_end = 22;

	
	getline(ifs, line);
	istringstream iss(line);
	getline(iss, value, ';');
	int header_index = 0;
	int find = value.find("Em");
	if(find == 0) // Header
	{
		n_lines -= 1;			 // Substract the heather line from the count
		head_ruta_fitxer = -1;
		head_nom_fitxer = -1;
		head_img_labels_ini = -1;
		head_img_labels_end = -1;
		head_features_end = -1;
		M = Mat(n_lines, Constants::NUM_CLASSIFIERS, CV_32SC1);
		F = Mat(n_lines, Constants::NUM_FEATURES, CV_32F);
		do  { 
			if (strcmp(value.c_str(), "EmRutaFitxer") == 0)
				head_ruta_fitxer = header_index;
			if(strcmp(value.c_str(),"EmNomFitxer") == 0)
				head_nom_fitxer = header_index;
			if(strcmp(value.c_str(),"EmBorrosa") == 0)
				head_img_labels_ini = header_index;
			if (strcmp(value.c_str(), "dedo") == 0)
			{
				head_img_labels_end = header_index;
			}
			if (strcmp(value.c_str(), ">.9") == 0)
			{	
				head_features_end = header_index;
			}
			header_index++;
		} while (getline(iss, value, ';'));
		getline(ifs, line);
	}
	do
	{
		if (line.empty())
		continue; 

		istringstream iss(line);

		// Read Line
		int cnt = 0;
		while (getline(iss, value, ';')) { 
			try
			{
				if (cnt == head_ruta_fitxer) {
					string pathFitxer;
					if(imagesBasePath != NULL)
						pathFitxer = imagesBasePath + value;
					else
						pathFitxer = value;
					imgPaths.push_back(pathFitxer);
				}
				
				if(cnt == head_nom_fitxer) {
					// Image name
					// Store it into string *imgPaths
					imgFileNames.push_back(value);
				}

				if (head_img_labels_ini > 0)
				{

					if (cnt >= head_img_labels_ini && cnt < head_img_labels_end) {
						// Image Labels
						M.at<int>(lcnt, cnt - head_img_labels_ini) = atoi(value.c_str());
					}
				}
				if (head_img_labels_end > 0)
				{
					if (cnt >= head_img_labels_end && cnt <= head_features_end)
					{
						if (cnt == head_img_labels_end) //dedo
						{
							int dedo = atoi(value.c_str());
							for (int i = 0; i < 10; i++)
							{
								if (dedo == i+1)
									F.at<float>(lcnt, cnt - head_img_labels_end + i) = 1;
								else
									F.at<float>(lcnt, cnt - head_img_labels_end + i) = 0;
							}
						}
						else
						{
							F.at<float>(lcnt, cnt - head_img_labels_end + 9) = (float)atof(value.c_str());
						}
						
					}
				}
				++cnt;
			} 
			catch(...)
			{
				cout << "Línia:" << lcnt << endl;
			}
		}
		++lcnt;
		
	} while(getline(ifs, line));

	if (head_ruta_fitxer = -1 && imagesBasePath != NULL)
	{
		imgPaths.push_back(imagesBasePath);
	}
	LabelsAndFeaturesData ret = { M, imgFileNames, imgPaths, F};

	*out = ret;
}
*/

void Utils::countLines(int* out, const char *path, bool verbose) {
	int number_of_lines = 0;
    string line;
    ifstream myfile(path);

	if(!myfile.is_open())
	{
		throwError((string)"ERROR: file " + path + " could not be opened. Is the path okay?");
	}
	clock_t begin = clock();
	while (getline(myfile, line))
	{

		++number_of_lines;
		if (number_of_lines % 10000 == 0 && verbose)
		{ 
			clock_t end = clock();
			cout << number_of_lines << " lines... " << double(end - begin) / CLOCKS_PER_SEC << " secs." << endl;
			begin = clock();
		}
	}
        

	
	myfile.close();

    *out = number_of_lines;
}

bool Utils::has_suffix(const string &str, const string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void Utils::getFileNameFromPath(string* output, const string path)
{
	int index = path.find_last_of("/") + 1;
	*output = path.substr(index);
}

int Utils::LoadCSV(CsvData* out, LoadCsvParams params)
{
	CsvData tmp;
	int nLines;

	if (verbose)
		cout << "Counting file lines... " << endl;
	countLines(&nLines, params.csvFile);
	if (verbose)
		cout << "Counting file lines completed: " << nLines << endl;

	ifstream ifs(params.csvFile, ifstream::in);

	if (!ifs.is_open())
		return -1;
	int cont;
	string line;
	string value;
	istringstream iss;
	if (params.withHeaders)
	{
		tmp.body = Mat(nLines - 1, params.globalRange.end - params.globalRange.begin, CV_32F);
		getline(ifs, line);
		iss = istringstream(line);

		cont = 0;

		while (cont < params.globalRange.begin)
		{
			getline(iss, value, params.separator);
			cont++;
		}

		for (int j = params.globalRange.begin; j < params.globalRange.end; j++)
		{
			getline(iss, value, params.separator);
			tmp.headers.push_back(value);
		}
	}
	else
		tmp.body = Mat(nLines, params.globalRange.end - params.globalRange.begin, CV_32F);

	for (int i = 0; i < tmp.body.rows; i++)
	{
		getline(ifs, line);
		iss = istringstream(line);
		cont = 0;
		string filePath;
		while (cont < params.globalRange.begin)
		{
			getline(iss, value, params.separator);
			if (cont == params.folderNameIndex)
			{
				filePath = params.baseImgPath + value;
				filePath = filePath + "/";
			}
			
			if (cont == params.fileNameIndex)
			{
				tmp.fileNames.push_back(filePath + value);
			}
			cont++;
		}
		// Read Line
		for (int j = 0; j < tmp.body.cols; j++)
		{
			getline(iss, value, params.separator);
			tmp.body.at<float>(i, j) = (float)atof(value.c_str());
			if (strcmp(value.c_str(), "true") == 0)
				tmp.body.at<float>(i, j) = 1.;
			if (strcmp(value.c_str(), "false") == 0)
				tmp.body.at<float>(i, j) = -1.;
		}
		if (verbose && tmp.body.rows > 10 && i % (tmp.body.rows / 10) == 0)
			cout << (i > 0 ? i * 100 / (tmp.body.rows) : 0) << "%" << endl;
	}
	*out = tmp;
	return 0;
}

void Utils::LoadFitDataFromFile(Mat* X_out, Mat* y_out, vector<string>* img_paths_out, LoadCsvParams impData)
{
	CsvData csv_data;
	LoadCSV(&csv_data, impData);
	
	if(impData.yRange != NULL)
	{
		if (impData.yRange->end == impData.yRange->begin)
			*y_out = csv_data.body.col(impData.yRange->begin);
		else
			*y_out = csv_data.body.colRange(impData.yRange->begin, impData.yRange->end);
	}
	Mat digFingers;
	if (impData.XRange != NULL) {
		csv_data.body = csv_data.body.colRange(impData.XRange->begin, impData.XRange->end);
		if (impData.fingerFieldName != NULL) {
			vector<string>::iterator it;
			it = find(csv_data.headers.begin(), csv_data.headers.end(), impData.fingerFieldName);
			int posDedo = (int)distance(csv_data.headers.begin(), it) - impData.XRange->begin;
			DigitalizeFingers(&digFingers, csv_data.body.col(posDedo));

			Mat output = csv_data.body.colRange(0, posDedo);
			if (output.cols == 0)
				output = csv_data.body.colRange(posDedo + 1, csv_data.body.cols);
			else
				hconcat(output, csv_data.body.colRange(posDedo + 1, csv_data.body.cols), output);
			hconcat(digFingers, output, output);
			*X_out = output;
		}
		else {
			*X_out = csv_data.body;
		}
	}

	*img_paths_out = csv_data.fileNames;
}

void Utils::calculateTimeLeft(long* timeLeft, long lastElapsed, int n, int i)
{
	*timeLeft = lastElapsed * (n - i);
}

void Utils::convertTime(string* output, long millis)
{
	string tmp;
	if (millis > 999)
	{
		long seconds = millis / 1000;
		tmp = to_string(millis % 1000) + "ms";
		if (seconds > 59)
		{
			long minutes = seconds / 60;
			tmp = to_string(seconds % 60) + "s and " + tmp;
			if (minutes > 59)
			{
				long hours = minutes / 60;
				tmp = to_string(minutes % 60) + "m," + tmp;
				if (hours > 23)
				{
					long days = hours / 24;
					tmp = to_string(hours % 23) + "h," + tmp;
					tmp = to_string(days) + "d," + tmp;
				}
				else
				{
					tmp = to_string(hours) + "h," + tmp;
				}
			}
			else
			{
				tmp = to_string(minutes) + "m," + tmp;
			}
		}
		else
		{
			tmp = to_string(seconds) + "s and " + tmp;
		}
	}
	else
	{
		tmp = to_string(millis) + "ms";
	}
	*output = tmp;
}

int Utils::DigitalizeFingers(Mat* output, const Mat onlyFingerNumber)
{
	Mat tmp = Mat(onlyFingerNumber.rows, 10, onlyFingerNumber.type());
	try
	{
		for (int i = 0; i < tmp.rows; i++)
		{
			int nFinger = (int)onlyFingerNumber.at<float>(i, 0);
			for (int j = 0; j < tmp.cols; j++)
			{
				if (j + 1 != nFinger)
					tmp.at<float>(i, j) = 0;
				else
					tmp.at<float>(i, j) = 1;
			}
		}
	}
	catch (...)
	{
		return -1;
	}
	*output = tmp;
	return 0;
}