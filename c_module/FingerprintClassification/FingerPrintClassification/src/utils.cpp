#include <utils.h>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"

using namespace std;

void throwError(std::string error) {
	std::string err = error;
	std::cerr << err << std::endl;
	throw std::exception(err.c_str());
}

/**************************************************************************
*								  readCSV
*								  -------
*		path  : Path to csv file.
*		head  : Boolean to determine whether the csv file has heather or not.
*
***************************************************************************/
LabelsAndFeaturesData readCSV(const char *path){

	// Count the number of lines in the file to allocate memory.
	int n_lines = countLines(path);
	if(n_lines == 0)
		throwError("ERROR: CSV " + (std::string)path + " contains no data");
	// Use string instead of char*
	std::string spath(path);

	// Open CSV file
	std::ifstream ifs(path, std::ifstream::in);
	if (!ifs.is_open()) {
		throwError("ERROR: file " + spath + " could not be opened. Is the path okay?");
	}

	std::string line;

	// Read File
	std::string value;
	//std::string dir = spath.substr(0, spath.find_last_of("\\/"));
	
	cv::Mat M(n_lines, Constants::NUM_CLASSIFIERS, CV_32SC1);
	
	cv::Mat F(n_lines, Constants::NUM_FEATURES, CV_32F);
	
	std::vector<std::string> imgPaths(0);
	
	int lcnt = 0;
	int head_nom_fitxer = 2;
	int head_img_labels_ini = 3;
	int head_img_labels_end = 9;

	
	std::getline(ifs, line);
	std::istringstream iss(line);
	std::getline(iss, value, ';');
	int header_index = 0;
	int find = value.find("Em");
	if(find == 0) // Header
	{
		n_lines -= 1;			 // Substract the heather line from the count
		M = cv::Mat(n_lines, Constants::NUM_CLASSIFIERS, CV_32SC1);
		F = cv::Mat(n_lines, Constants::NUM_FEATURES, CV_32F);
		do  { 
			if(strcmp(value.c_str(),"EmNomFitxer") == 0)
				head_nom_fitxer = header_index;
			if(strcmp(value.c_str(),"EmBorrosa") == 0)
				head_img_labels_ini = header_index;
			if(strcmp(value.c_str(),"EmDefectuosa") == 0)
				head_img_labels_end = header_index;
			header_index++;
		} while (std::getline(iss, value, ';'));
		std::getline(ifs, line);
	}
	do
	{
		if (line.empty())
		continue; 

		std::istringstream iss(line);

		// Read Line
		int cnt = 0;

		while (std::getline(iss, value, ';')) { 
			try
			{
				if(cnt == head_nom_fitxer) {
					// Image name
					// Store it into std::string *imgPaths
					imgPaths.push_back(value);
				}

				if(cnt >= head_img_labels_ini && cnt <= head_img_labels_end) {
					// Image Labels
					M.at<int>(lcnt,cnt-head_img_labels_ini) = atoi(value.c_str());
				}
			
				if(cnt > head_img_labels_end)
				{
					F.at<float>(lcnt,cnt-head_img_labels_end - 1) = (float)atof(value.c_str());
				}
			
				++cnt;
			} 
			catch(...)
			{
				std::cout << "Línia:" << lcnt << std::endl;
			}
		}
		++lcnt;
	} while(std::getline(ifs, line));


	LabelsAndFeaturesData ret = { M, imgPaths, F};

	return ret;
}

int countLines(const char *path) {
   unsigned int number_of_lines = 0;
    FILE *infile;
	fopen_s(&infile, path, "r");
    int ch;

	if(NULL == infile)
	{
		throwError((std::string)"ERROR: file " + path + " could not be opened. Is the path okay?");
	}
		

    while (EOF != (ch=getc(infile)))
        if ('\n' == ch)
            ++number_of_lines;

	fclose(infile);

    return number_of_lines;
}

cv::Mat oneVsAll(cv::Mat labels, int tar_class){
	cv::Mat res(labels.cols, 1, labels.type());

	for(int i = 0; i<labels.cols; i++){
		// Set 0 if the class is not tar_class
		// Set 1 if the class is tar_class
		if(labels.at<int>(tar_class, i) != 1)
			res.at<int>(i, 0) = 0;
		else
			res.at<int>(i, 0) = 1;
	}
	return res;
}

void exportFileFeatures(cv::Mat trainSamples, std::vector<std::string> imgPaths, const char* outFile)
{
	/*char path_buffer[_MAX_PATH];
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
				
	errno_t err;
				
	err = _splitpath_s(outPath,drive,_MAX_DRIVE,dir,_MAX_DIR,fname,_MAX_FNAME,ext,_MAX_EXT);
	if(err != 0)
	{
		throwError("Couldn't split path:" + (std::string)outPath);
	}*/
	std::string fileName = outFile;
	std::ofstream myfile (fileName);
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
	std::cout << "n_bins:" << prop.n_bins << std::endl;
	std::cout << "rad_grad:" << prop.rad_grad << std::endl;
	std::cout << "rad_dens:" << prop.rad_dens << std::endl;
	std::cout << "rad_entr:" << prop.rad_entr << std::endl;
	std::cout << "max_depth:" << prop.max_depth << std::endl;
	std::cout << "min_samples_count:" << prop.min_samples_count << std::endl;
	std::cout << "max_categories:" << prop.max_categories << std::endl;
	std::cout << "max_num_of_trees_in_forest:" << prop.max_num_of_trees_in_forest << std::endl;

	std::cout << "TOTAL_FEATURES:" << Constants::TOTAL_FEATURES << std::endl;
	std::cout << "NUM_ROW_SEGMENTS:" << Constants::NUM_ROW_SEGMENTS << std::endl;
	std::cout << "NUM_COL_SEGMENTS:" << Constants::NUM_COL_SEGMENTS << std::endl;
	std::cout << "NUM_CLASSIFIERS:" << Constants::NUM_CLASSIFIERS << std::endl;
	std::cout << "NUM_FEATURES:" << Constants::NUM_FEATURES << std::endl;
}

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void loadNormalization(Mat* norMat, const char* normFile)
{
	std::ifstream ifs(normFile, std::ifstream::in);
	if (!ifs.is_open()) {
		throwError((std::string)"ERROR: file " + normFile + " could not be opened. Is the path okay?");
	}

	std::string line;
	std::string value;
	cv::Mat ret = cv::Mat(Constants::TOTAL_FEATURES,2,CV_32F);
	
	for(int i = 0; i < ret.rows; i++)
	{
		std::getline(ifs, line);
		if (line.empty())
			continue; 
		std::istringstream iss(line);
		// Read Line
		for(int j = 0; j < ret.cols; j++)
		{
			std::getline(iss, value, ' ');
			ret.at<float>(i,j) = (float)atof(value.c_str());
		}
	}
	*norMat = ret;
}

void saveNormalization(const Mat* norMat, const char* normFile)
{	
	ofstream file;
	file.open(normFile);
	
	for (int i = 0; i < norMat->rows; i++)
	{
		file << norMat->at<float>(i,0) << ' ' << norMat->at<float>(i,1) << endl;
	}

	file.close();
}

cv::Mat importFileFeatures(const char* c_path_normalized, bool verbose, const int total_features)
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

	std::string line;
	std::string value;
	cv::Mat ret = cv::Mat(n_lines,total_features,CV_32F);
	int percent = n_lines / 10;
	if(verbose)
		std::cout << "Loading data..." << std::endl;
	for(int i = 0; i < ret.rows; i++)
	{
		std::getline(ifs, line);
		if (line.empty())
			continue; 
		std::istringstream iss(line);
		// Read Line
		for(int j = 0; j < ret.cols; j++)
		{
			std::getline(iss, value, ',');
			if(has_suffix(value,".png"))
			{
				if(verbose && i % percent == 0)
					std::cout << "imagen[" << i << "]" << value << std::endl;
				//Useless fileName header
				j--;
				continue;
			}
			ret.at<float>(i,j) = (float)atof(value.c_str());
		}
	}


	return ret;
}

cv::Mat createNormalizationFile(const char* outPath, cv::Mat trainSamples) 
{
	//NORMALIZE
	// Write the mean and std into a file as part of the model
	char fname[10000];
		
	sprintf_s(fname, "%snormalization.csv", outPath);
		
	std::ofstream file;
	file.open(fname);
	cv::Mat temp1, temp2, mean, std, norm_i,ret;
	ret = cv::Mat(trainSamples.size(), trainSamples.type());
	//for (int i = trainSamples.cols - 1; i >= 0; i--)
	for (int i = 0; i < trainSamples.cols; i++)
	{
		cv::meanStdDev(trainSamples.col(i), mean, std);
		cv::subtract(trainSamples.col(i), mean, temp1);
		cv::divide(temp1, std, temp2);
		norm_i = ret.colRange(i, i+1);
		temp2.copyTo(norm_i);
		file << mean.at<double>(0,0) << ' ' << std.at<double>(0,0) << std::endl;
	}
	
	file.close();
	return ret;
}

cv::Mat readTrainedMeanStd(const char* normalizationFilePath,cv::Mat sample) 
{
	//NORMALIZE
	// Read train mean and std to normalize the test mean
	//TODO: Normalize test
	char fname[10000];
	sprintf_s(fname, "%snormalization.csv", normalizationFilePath);
	std::ifstream file;
	file.open(fname);

	// initialize matrices
	cv::Mat normSample = cv::Mat(sample.size(), sample.type());
	cv::Mat temp1, temp2, norm_i;
	std::string line;
	for (int i = sample.cols - 1; i >= 0; i--){
		// Read mean and std from csv line
		std::getline(file, line);
		std::istringstream iss(line);
		float a, b;
		if (!(iss >> a >> b)) 
			break;// error
		/*else {
			cv::Mat mean(a);
			cv::Mat std(b);
		}*/

		cv::subtract(sample.col(i), a, temp1);
		cv::divide(temp1, b, temp2);
		norm_i = normSample.colRange(i, i+1);
		temp2.copyTo(norm_i);
		//mean.release();
		//std.release();
	}

	file.close();
	return normSample;
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
cv::Mat CropImage(int row, int col, const cv::Mat* img)
{
	int row_split = img->rows / Constants::NUM_ROW_SEGMENTS;
	int col_split = img->cols / Constants::NUM_COL_SEGMENTS;
	int mod_row_split = img->rows % Constants::NUM_ROW_SEGMENTS;
	int mod_col_split = img->cols % Constants::NUM_COL_SEGMENTS;

	int start_row = row*row_split;
	int end_row = (row + 1)*row_split;
	int start_col = col*col_split;
	int end_col = (col + 1)*col_split;

	if (end_row == img->rows - mod_row_split)
		end_row += mod_row_split;

	if (end_col == img->cols - mod_col_split)
		end_col += mod_col_split;

	cv::Rect rect = cv::Rect(start_col, start_row, end_col - start_col, end_row - start_row);

	return cv::Mat((*img)(rect));
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
cv::Mat** GetImageRegions(const cv::Mat *img) {

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
