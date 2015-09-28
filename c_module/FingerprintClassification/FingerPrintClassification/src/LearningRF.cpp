#include "LearningRF.h"
#include <FpFeaturesLibrary.h>
#include <time.h>
#include <fstream>
#include "utils.h"

using namespace cv;
using namespace std;

void LearnData::write(const char* outPath)
{
	string fileName = outPath;
	fileName += "/normalization.csv";
	
	ofstream myfile(fileName);
	if (myfile.is_open())
	{
		for (int i = 0; i < normalization.rows; i++)
		{
			for (int j = 0; j < normalization.cols; j++)
				myfile << " " << normalization.at<float>(i, j);
			myfile << endl;
		}
		myfile.close();
	}
	else
	{
		throwError("Unable to open file " + fileName);
	}

	for (int m = 0; m < Constants::NUM_CLASSIFIERS; m++)
	{
		char fileName[10000];
		sprintf(fileName, "%smodel_%d.xml", outPath, m);
		rtrees->save((const char*)fileName);
	}
}

LearningRF::LearningRF()
{
	prop = new Properties();
}

LearningRF::~LearningRF()
{
	delete prop;
}

void LearningRF::ImageExtraction(const Mat* img, Mat* output)
{
	Mat ret;
	diferentiate_img(&ret, img);
	
	cv::Mat** regions = GetImageRegions(img);
	
	for (int i = Constants::NUM_ROW_SEGMENTS - 1; i >= 0; i--)
	{
		for (int j = Constants::NUM_COL_SEGMENTS - 1; j >= 0; j--)
		{
			cv::Mat *in = new Mat(regions[i][j]);
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

void LearningRF::Extract(const vector<Mat*>* imgData, Mat* rawFeatures)
{
	Mat tmp = Mat();

	for (unsigned int i = 0; i < imgData->size(); i++)
	{
		clock_t time_a = clock();
		cv::Mat hist;
		ImageExtraction((*imgData)[i], &hist);
		// Join features
		if (tmp.rows == 0)
			tmp = hist;
		else
			cv::vconcat(tmp, hist, tmp);
		
		clock_t time_b = clock();
		if (prop->verbose)
			cout << "extract " << (i+1) << " of " << imgData->size() << ". length: [" << tmp.rows << "," << tmp.cols << "] time: " << (long)(time_b - time_a) << endl;
	}
	
	*rawFeatures = tmp;
}

void LearningRF::CreateNorm(const cv::Mat* input, cv::Mat* normalization, cv::Mat* output)
{
	Mat nzation;
	Mat nzed;

	cv::Mat temp1, temp2, mean, std, norm_i;
	nzed = cv::Mat(input->size(), input->type());
	nzation = cv::Mat(input->cols, 2, input->type());
	for (int i = 0; i < input->cols; i++)
	{
		cv::meanStdDev(input->col(i), mean, std);
		//mean.convertTo(mean,CV_32F);
		//std.convertTo(std,CV_32F);
		cv::subtract(input->col(i), mean, temp1);
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

void LearningRF::Fit(CvRTrees** ret, const Mat* labels, const Mat* normFeatures)
{
	/****************************************************/
	/*					  Train RF						*/
	/****************************************************/

	// Construct the classifier and set the parameters
	//CvRTrees  rtrees[Constants::NUM_CLASSIFIERS][1];
	CvRTrees* rtrees = new CvRTrees[Constants::NUM_CLASSIFIERS];
	// Construct the classifier and set the parameters
	//CvRTrees** rtrees;
	//allocateRtrees(&rtrees,Constants::NUM_CLASSIFIERS,1);
	float priors[] = { 1, 1, 1, 1, 1, 1 };
	CvRTParams  params(prop->max_depth,		// max_depth,
		prop->min_samples_count,			// min_sample_count,
		0.f,								// regression_accuracy,
		false,								// use_surrogates,
		prop->max_categories,				// max_categories,
		priors,								// priors,
		false,								// calc_var_importance,
		prop->nactive_vars,					// nactive_vars,
		prop->max_num_of_trees_in_forest,	// max_num_of_trees_in_the_forest,
		0,									// forest_accuracy,
		CV_TERMCRIT_ITER					// termcrit_type
		);

	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis
	cv::Mat var_type = cv::Mat(Constants::TOTAL_FEATURES + 1, 1, CV_8U);
	var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical
	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
	var_type.at<uchar>(Constants::TOTAL_FEATURES, 0) = CV_VAR_CATEGORICAL;
	/*cv::FileStorage file_norm("normTS.txt", cv::FileStorage::WRITE);
	file_norm << "normTS" << normTS;*/
	for (int i = 0; i < Constants::NUM_CLASSIFIERS; i++){

		cv::Mat trainClass_i;
		//Prepare trainClasses for the oneVsAll classification strategy
		trainClass_i = labels->col(i);
		/*if (prop->verbose)
		{
			std::cout << "Columna_Labels[" << i << "]:";
			for (int i = 0; i < 10; i++)
				std::cout << trainClass_i.at<int>(i, 0) << " ";
			std::cout << std::endl;
		}*/
		// Fit the classifier with the training data
		/*cv::FileStorage file_train("trainClass" + std::to_string((long double)i) + ".txt", cv::FileStorage::WRITE);
		file_train << "trainClass_i" << trainClass_i;*/
		if (prop->verbose)
			std::cout << "Training(" << i << ")...";
		clock_t start = clock();
		rtrees[i].train(*normFeatures, CV_ROW_SAMPLE, trainClass_i);
		//rtrees[i].train(*normFeatures, CV_ROW_SAMPLE, trainClass_i, Mat(), Mat(), var_type, Mat(), params);
		if (prop->verbose)
			std::cout << "time:" << clock() - start << "ms" << std::endl;
	}
	*ret = rtrees;
}

void LearningRF::Normalize(const Mat* input, Mat* output, const Mat* normalization)
{
	// initialize matrices
	cv::Mat normSample = cv::Mat(input->size(), input->type());
	cv::Mat temp1, temp2, norm_i;
	std::string line;
	for (int i = 0; i < input->cols; i++)
	{
		// Read mean and std from normalization Mat
		Mat row_i = normalization->row(i);

		float a, b;
		a = row_i.at<float>(0, 0);
		b = row_i.at<float>(0, 1);

		cv::subtract(input->col(i), a, temp1);
		cv::divide(temp1, b, temp2);
		norm_i = normSample.colRange(i, i + 1);
		temp2.copyTo(norm_i);
	}

	*output = normSample;
}

void LearningRF::Predict(float** ret, CvRTrees* rtrees, const Mat* normFeatures)
{
	float *prediction = new float[Constants::NUM_CLASSIFIERS];

	// Initialice rtree;
	for (int i = 0; i<Constants::NUM_CLASSIFIERS; i++){

		prediction[i] = rtrees[i].predict_prob(*normFeatures);
	}

	*ret = prediction;
}
