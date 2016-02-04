#include "ImgTools.h"
#include "FpFeaturesLibrary.h"
#include <time.h>
#include "opencv2\highgui\highgui.hpp"
#include "utils.h"
#include <fstream>

ImgTools::ImgTools()
{
	prop = new ImgProcessingProperties();
}

ImgTools::~ImgTools()
{
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
void ImgTools::CropImage(Mat* out, int row, int col, const Mat img)
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

	*out = Mat((img)(rect));
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
void ImgTools::GetImageRegions(Mat*** out, const Mat img) {

	cv::Mat** ret = new Mat*[Constants::NUM_ROW_SEGMENTS];
	for (int i = 0; i < Constants::NUM_ROW_SEGMENTS; ++i)
		ret[i] = new Mat[Constants::NUM_COL_SEGMENTS];

	for (int i = 0; i < Constants::NUM_ROW_SEGMENTS; i++) {
		for (int j = 0; j < Constants::NUM_COL_SEGMENTS; j++) {
			Mat cropped;
			CropImage(&cropped, i, j, img);
			ret[i][j] = Mat(cropped);
		}
	}
	*out = ret;
}

void ImgTools::ImageExtraction(const Mat img, Mat* output)
{
	Mat ret;
	diferentiate_img(&ret, img);


	cv::Mat** regions;
	GetImageRegions(&regions, img);

	for (int i = Constants::NUM_ROW_SEGMENTS - 1; i >= 0; i--)
	{
		for (int j = Constants::NUM_COL_SEGMENTS - 1; j >= 0; j--)
		{
			cv::Mat in = regions[i][j];
			cv::Mat out_grad;
			hist_grad(&out_grad, in, prop->rad_grad, prop->n_bins);
			cv::Mat out_dens;
			hist_density(&out_dens, in, prop->rad_dens, prop->n_bins);
			cv::Mat out_hough;
			hist_hough(&out_hough, in, prop->n_bins);
			cv::Mat out_entropy;
			hist_entropy(&out_entropy, in, prop->rad_entr, prop->n_bins);
			// Join histograms
			cv::hconcat(out_hough, ret, ret);
			cv::hconcat(out_entropy, ret, ret);
			cv::hconcat(out_grad, ret, ret);
			cv::hconcat(out_dens, ret, ret);
			//Utils::saveMatToCSV<float>(ret, "hist.csv");
		}
	}

	for (int i = 0; i < Constants::NUM_ROW_SEGMENTS; ++i) {
		delete[] regions[i];
	}
	delete[] regions;

	cv::Mat out_grad;
	hist_grad(&out_grad, img, prop->rad_grad, prop->n_bins);
	cv::Mat out_dens;
	hist_density(&out_dens, img, prop->rad_dens, prop->n_bins);
	cv::Mat out_hough;
	hist_hough(&out_hough, img, prop->n_bins);
	cv::Mat out_entropy;
	hist_entropy(&out_entropy, img, prop->rad_entr, prop->n_bins);
	// Join histograms	
	cv::hconcat(out_hough, ret, ret);
	cv::hconcat(out_entropy, ret, ret);
	cv::hconcat(out_grad, ret, ret);
	cv::hconcat(out_dens, ret, ret);

	*output = ret;
}

/********************************************************************/
/* Extraction of a set of images									*/
/********************************************************************/
void ImgTools::Extract(const vector<string> imgPaths, Mat* rawFeatures)
{
	Mat tmp = Mat();

	string base = "";

	for (unsigned int i = 0; i < imgPaths.size(); i++)
	{
		clock_t time_a = clock();

		Mat in;
		string path = imgPaths[i];


		string fileName;
		Utils::getFileNameFromPath(&fileName, path);
		in = imread(path, IMREAD_GRAYSCALE);

		if (in.rows == 0)
		{
			if (Utils::verbose)
			{
				cout << "[" << fileName << "]" << (i + 1) << " of " << imgPaths.size() << ". Not Found!" << endl;
				ofstream myfile("extractNF.txt", ios_base::app);
				if (myfile.is_open())
				{
					myfile << path << endl;
					myfile.close();
				}
				else cout << "Unable to open file";
			}
			continue;
		}
		Mat hist;
		ImageExtraction(in, &hist);
		// Join features
		if (tmp.rows == 0)
			tmp = hist;
		else
			cv::vconcat(tmp, hist, tmp);

		clock_t time_b = clock();
		if (Utils::verbose)
		{
			long lastElapsed = (long)(time_b - time_a);
			cout << "[" << fileName << "]" << (i + 1) << " of " << imgPaths.size();
			long timeLeft;
			Utils::calculateTimeLeft(&timeLeft, lastElapsed, imgPaths.size(), i);
			string showTime;
			Utils::convertTime(&showTime, timeLeft);
			cout << ".Left:" << showTime << endl;
		}
	}

	*rawFeatures = tmp;
}

void ImgTools::ImportImageFeatures(vector<string>* fNames, Mat* normData, const char* c_path_normalized, const int total_features)
{
	char* path = (char*)c_path_normalized;

	if (Utils::verbose)
		std::cout << "Counting lines..." << std::endl;
	int n_lines;
	Utils::countLines(&n_lines, path, Utils::verbose);
	if (Utils::verbose)
		std::cout << "{" << n_lines << "}" << std::endl;


	// Open CSV file
	ifstream ifs(path, ifstream::in);
	if (!ifs.is_open()) {
		Utils::throwError((string)"ERROR: file " + path + " could not be opened. Is the path okay?");
	}
	vector<string> imgFileNames(0);
	string line;
	string value;
	Mat ret = Mat(n_lines, total_features, CV_32F);
	int percent = n_lines / 10;
	if (Utils::verbose)
		std::cout << "Loading data..." << std::endl;
	for (int i = 0; i < ret.rows; i++)
	{
		getline(ifs, line);
		if (line.empty())
			continue;
		istringstream iss(line);
		// Read Line
		for (int j = 0; j < ret.cols; j++)
		{
			getline(iss, value, ',');
			if (Utils::has_suffix(value, ".png"))
			{
				if (Utils::verbose && i % percent == 0)
					cout << "imagen[" << i << "]" << value << endl;
				imgFileNames.push_back(value);
				j--;
				continue;
			}
			ret.at<float>(i, j) = (float)atof(value.c_str());
		}
	}
	*normData = ret;
	*fNames = imgFileNames;
}

void ImgTools::ExportImageFeatures(Mat trainSamples, vector<string> imgPaths, const char* outFile)
{
	string fileName = outFile;
	ofstream myfile(fileName);
	if (myfile.is_open())
	{
		for (int i = 0; i < trainSamples.rows; i++)
		{
			string fileName;
			Utils::getFileNameFromPath(&fileName, imgPaths[i]);
			myfile << fileName;
			for (int j = 0; j < trainSamples.cols; j++)
				myfile << "," << trainSamples.at<float>(i, j);
			myfile << "\n";
		}
		myfile.close();
	}
	else
	{
		Utils::throwError("Unable to open file " + fileName);
	}
}