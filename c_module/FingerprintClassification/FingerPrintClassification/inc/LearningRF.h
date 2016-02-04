#include "opencv2\ml\ml.hpp"
#include "utils.h"
#pragma once

using namespace cv;
using namespace std;

struct RFProperties
{
	int max_depth; // Max depth of the trees in the Random Forest.
	int min_samples_count; // Min samples needed to split a leaf.
	int max_categories; // Max number of categories.
	int max_num_of_trees_in_forest; // Max number of trees in the forest.
	int nactive_vars; // nactive_vars,

	RFProperties() {
		max_depth = 16;
		min_samples_count = 2;
		max_categories = 3;
		max_num_of_trees_in_forest = 10;
		nactive_vars = 0;
	};
	friend std::ostream& operator<<(std::ostream& os, const RFProperties& prop);
};

class LearningRF
{
public:
	RFProperties* prop;
	LearningRF();
	~LearningRF();
	void Fit(CvRTrees**, const Mat, const Mat);
	void Predict(float**, CvRTrees*, const Mat);
	static void printParamsRF(const RFProperties&);
	void allocateRtrees(CvRTrees***, const int, const int);
	void releaseRTrees(CvRTrees**, const int, const int);
};

