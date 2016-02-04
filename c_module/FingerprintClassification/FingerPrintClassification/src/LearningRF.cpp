#include "LearningRF.h"
#include <FpFeaturesLibrary.h>
#include <time.h>
#include <fstream>
#include "utils.h"
#include "opencv2\highgui\highgui.hpp"

using namespace cv;
using namespace std;

LearningRF::LearningRF()
{
	prop = new RFProperties();
}

LearningRF::~LearningRF()
{
	delete prop;
}

void LearningRF::Fit(CvRTrees** ret, const Mat labels, const Mat normFeatures)
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
		trainClass_i = labels.col(i);
		
		if (Utils::verbose)
			std::cout << "Training(" << i << ")...";
		clock_t start = clock();
		rtrees[i].train(normFeatures, CV_ROW_SAMPLE, trainClass_i);
		//rtrees[i].train(*normFeatures, CV_ROW_SAMPLE, trainClass_i, Mat(), Mat(), var_type, Mat(), params);
		if (Utils::verbose)
			std::cout << "time:" << clock() - start << "ms" << std::endl;
	}
	*ret = rtrees;
}

void LearningRF::Predict(float** ret, CvRTrees* rtrees, const Mat normFeatures)
{
	float *prediction = new float[Constants::NUM_CLASSIFIERS];

	// Initialice rtree;
	for (int i = 0; i < Constants::NUM_CLASSIFIERS; i++){

		prediction[i] = rtrees[i].predict_prob(normFeatures);
	}

	*ret = prediction;
}

void LearningRF::printParamsRF(const RFProperties& prop)
{
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

void LearningRF::allocateRtrees(CvRTrees*** data, const int rows, const int cols)
{
	CvRTrees** rtrees = new CvRTrees*[rows];
	for (int i = 0; i < rows; ++i)
		rtrees[i] = new CvRTrees[cols];
	*data = rtrees;
}

void LearningRF::releaseRTrees(CvRTrees** matrix, const int rows, const int cols)
{
	for (int i = 0; i < rows; ++i)
		delete[] matrix[i];
	delete[] matrix;
}