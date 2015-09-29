#include <utils.h>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

using namespace std;
using namespace cv;

void throwError(string error) {
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
***************************************************************************/
LabelsAndFeaturesData readCSV(const char *path, const char* imagesBasePath){

	// Count the number of lines in the file to allocate memory.
	int n_lines = countLines(path);
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

	LabelsAndFeaturesData ret = { M, imgFileNames, imgPaths, F};

	return ret;
}

int countLines(const char *path) {
	int number_of_lines = 0;
    string line;
    ifstream myfile(path);

	if(NULL == myfile)
	{
		throwError((string)"ERROR: file " + path + " could not be opened. Is the path okay?");
	}

    while (getline(myfile, line))
        ++number_of_lines;
	
	myfile.close();

    return number_of_lines;
}

void exportFileFeatures(cv::Mat trainSamples, std::vector<std::string> imgPaths, const char* outFile)
{
	string fileName = outFile;
	ofstream myfile (fileName);
	if (myfile.is_open())
	{
		for(int i = 0; i < trainSamples.rows; i++)
		{
			myfile << imgPaths[i];
			for(int j = 0; j < trainSamples.cols; j++)
				myfile << "," << trainSamples.at<float>(i,j);
			myfile << "\n";
		}
		myfile.close();
	}
	else
	{
		throwError("Unable to open file " + fileName);
	}
}

void printParamsRF(const Properties& prop)
{
	cout << "n_bins:" << prop.n_bins << endl;
	cout << "rad_grad:" << prop.rad_grad << endl;
	cout << "rad_dens:" << prop.rad_dens << endl;
	cout << "rad_entr:" << prop.rad_entr << endl;
	cout << "max_depth:" << prop.max_depth << endl;
	cout << "min_samples_count:" << prop.min_samples_count << endl;
	cout << "max_categories:" << prop.max_categories << endl;
	cout << "max_num_of_trees_in_forest:" << prop.max_num_of_trees_in_forest << endl;

	cout << "TOTAL_FEATURES:" << Constants::TOTAL_FEATURES << endl;
	cout << "NUM_ROW_SEGMENTS:" << Constants::NUM_ROW_SEGMENTS << endl;
	cout << "NUM_COL_SEGMENTS:" << Constants::NUM_COL_SEGMENTS << endl;
	cout << "NUM_CLASSIFIERS:" << Constants::NUM_CLASSIFIERS << endl;
	cout << "NUM_FEATURES:" << Constants::NUM_FEATURES << endl;
}

bool has_suffix(const string &str, const string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void loadNormalization(Mat* norMat, const char* normFile)
{
	ifstream ifs(normFile, ifstream::in);
	if (!ifs.is_open()) {
		throwError((string)"ERROR: file " + normFile + " could not be opened. Is the path okay?");
	}

	string line;
	string value;
	Mat ret = cv::Mat(Constants::TOTAL_FEATURES,2,CV_32F);
	
	for(int i = 0; i < ret.rows; i++)
	{
		getline(ifs, line);
		if (line.empty())
			continue; 
		istringstream iss(line);
		// Read Line
		for(int j = 0; j < ret.cols; j++)
		{
			getline(iss, value, ' ');
			ret.at<float>(i,j) = (float)atof(value.c_str());
		}
	}
	*norMat = ret;
}

void saveNormalization(const Mat norMat, const char* normFile)
{	
	ofstream file;
	file.open(normFile);
	
	for (int i = 0; i < norMat.rows; i++)
	{
		file << norMat.at<float>(i,0) << ' ' << norMat.at<float>(i,1) << endl;
	}

	file.close();
}

Mat importFileFeatures(const char* c_path_normalized, bool verbose, const int total_features)
{
	char* path = (char*)c_path_normalized;

	if(verbose)
		std::cout << "Counting lines..." << std::endl;
	int n_lines = countLines(path);
	if(verbose)
		std::cout << "{" << n_lines << "}" << std::endl;


	// Open CSV file
	std::ifstream ifs(path, std::ifstream::in);
	if (!ifs.is_open()) {
		throwError((std::string)"ERROR: file " + path + " could not be opened. Is the path okay?");
	}

	string line;
	string value;
	Mat ret = cv::Mat(n_lines,total_features,CV_32F);
	int percent = n_lines / 10;
	if(verbose)
		std::cout << "Loading data..." << std::endl;
	for(int i = 0; i < ret.rows; i++)
	{
		getline(ifs, line);
		if (line.empty())
			continue; 
		istringstream iss(line);
		// Read Line
		for(int j = 0; j < ret.cols; j++)
		{
			getline(iss, value, ',');
			if(has_suffix(value,".png"))
			{
				if(verbose && i % percent == 0)
					cout << "imagen[" << i << "]" << value << endl;
				//Useless fileName header
				j--;
				continue;
			}
			ret.at<float>(i,j) = (float)atof(value.c_str());
		}
	}
	return ret;
}

void allocateRtrees(CvRTrees*** data, const int rows, const int cols)
{
	CvRTrees** rtrees = new CvRTrees*[rows];
	for (int i = 0; i < rows; ++i)
		rtrees[i] = new CvRTrees[cols];
	*data = rtrees;
}

void releaseRTrees(CvRTrees** matrix, const int rows, const int cols) 
{
	for (int i = 0; i < rows; ++i)
		delete [] matrix[i];
	delete [] matrix;
}

/**************************************************************************
*						CropImage
*						----------------------
*		params->
*			row			: Row selected
*			col			: Col selected
*			img			: Reference to the image.
*		returns->
*			cv::Mat subset of img region(row,col)
*
***************************************************************************/
cv::Mat CropImage(int row, int col, const Mat img)
{
	int row_split = img.rows / Constants::NUM_ROW_SEGMENTS;
	int col_split = img.cols / Constants::NUM_COL_SEGMENTS;
	int mod_row_split = img.rows % Constants::NUM_ROW_SEGMENTS;
	int mod_col_split = img.cols % Constants::NUM_COL_SEGMENTS;

	int start_row = row*row_split;
	int end_row = (row + 1)*row_split;
	int start_col = col*col_split;
	int end_col = (col + 1)*col_split;

	if (end_row == img.rows - mod_row_split)
		end_row += mod_row_split;

	if (end_col == img.cols - mod_col_split)
		end_col += mod_col_split;

	cv::Rect rect = cv::Rect(start_col, start_row, end_col - start_col, end_row - start_row);

	return cv::Mat((img)(rect));
}

/**************************************************************************
*						GetImageRegions
*						----------------------
*		params->
*			img		 : Reference to the image.
*		returns->
*			array of cv::Mat with every region
*
***************************************************************************/
cv::Mat** GetImageRegions(const Mat img) {

	cv::Mat** ret = new cv::Mat*[Constants::NUM_ROW_SEGMENTS];
	for (int i = 0; i < Constants::NUM_ROW_SEGMENTS; ++i)
		ret[i] = new cv::Mat[Constants::NUM_COL_SEGMENTS];

	for (int i = 0; i < Constants::NUM_ROW_SEGMENTS; i++) {
		for (int j = 0; j < Constants::NUM_COL_SEGMENTS; j++) {
			ret[i][j] = CropImage(i, j, img);
		}
	}
	return ret;
}
