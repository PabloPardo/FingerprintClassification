#include <fstream>
#include <iostream>
#include "NorManagement.h"

using namespace std;

void NorManagement::CreateNorm(const Mat input, Mat* normalization, Mat* output)
{
	Mat nzation;
	Mat nzed;

	cv::Mat temp1, temp2, mean, std, norm_i;
	nzed = cv::Mat(input.size(), input.type());
	nzation = cv::Mat(input.cols, 2, input.type());
	for (int i = 0; i < input.cols; i++)
	{
		cv::meanStdDev(input.col(i), mean, std);
		//mean.convertTo(mean,CV_32F);
		//std.convertTo(std,CV_32F);
		cv::subtract(input.col(i), mean, temp1);
		cv::divide(temp1, std, temp2);
		norm_i = nzed.colRange(i, i + 1);
		temp2.copyTo(norm_i);

		nzation.at<float>(i, 0) = (float)mean.at<double>(0, 0);
		nzation.at<float>(i, 1) = (float)std.at<double>(0, 0);
	}

	*normalization = nzation;
	if (output != NULL)
		*output = nzed;
}
void NorManagement::Normalize(const Mat input, Mat* output, const Mat normalization)
{
	// initialize matrices
	cv::Mat normSample = cv::Mat(input.size(), input.type());
	cv::Mat temp1, temp2, norm_i;
	std::string line;
	for (int i = 0; i < input.cols; i++)
	{
		// Read mean and std from normalization Mat
		Mat row_i = normalization.row(i);

		float a, b;
		a = row_i.at<float>(0, 0);
		b = row_i.at<float>(0, 1);

		cv::subtract(input.col(i), a, temp1);
		cv::divide(temp1, b, temp2);
		norm_i = normSample.colRange(i, i + 1);
		temp2.copyTo(norm_i);
	}

	*output = normSample;
}
void NorManagement::LoadNormalization(Mat* norMat, const char* normFile, int nFeat)
{
	ifstream ifs(normFile, ifstream::in);
	if (!ifs.is_open()) {
		string err = (string)"ERROR: file " + normFile + " could not be opened. Is the path okay?";
		cerr << err << endl;
		return;
	}

	string line;
	string value;
	Mat ret = cv::Mat(nFeat, 2, CV_32F);

	for (int i = 0; i < ret.rows; i++)
	{
		getline(ifs, line);
		if (line.empty())
			continue;
		istringstream iss(line);
		// Read Line
		for (int j = 0; j < ret.cols; j++)
		{
			getline(iss, value, ' ');
			ret.at<float>(i, j) = (float)atof(value.c_str());
		}
	}
	*norMat = ret;
}
void NorManagement::SaveNormalization(const Mat norMat, const char* normFile)
{
	ofstream file;
	file.open(normFile);

	for (int i = 0; i < norMat.rows; i++)
	{
		file << norMat.at<float>(i, 0) << ' ' << norMat.at<float>(i, 1) << endl;
	}

	file.close();
}
