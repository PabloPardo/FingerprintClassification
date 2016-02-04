#include "AdaBoost.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <numeric>
#include "utils.h"
#include <iterator>

AdaBoost::AdaBoost()
{
	prop = new AdaBoostProperties();
}

AdaBoost::~AdaBoost()
{
}

using namespace std;

istream & operator>>(istream& str, Inequality& v) {
	unsigned int ineq = 0;
	if (str >> ineq)
		v = static_cast<Inequality>(ineq);
	return str;
}

istream& operator>>(istream& is, Stump& en)
{
	is >> en.alpha;
	is >> en.dim;
	is >> en.ineq;
	is >> en.thresh;
	return is;
}

ostream& operator<<(ostream& os, const Stump& en)
{
	os << en.alpha << " " << en.dim << " " << en.ineq << " " << en.thresh;
	return os;
}

void AdaBoost::StumpClassify(Mat* out, const Mat dataMatrix, int dim, float threshold, Inequality ineq)
{
	Mat ret_array = Mat(dataMatrix.rows, 1, CV_32F, 1.);
	switch (ineq)
	{
	case BIGGER_THAN:
		for (int i = 0; i < dataMatrix.rows; i++)
		{
			float val = dataMatrix.at<float>(i, dim);
			if (val > threshold)
				ret_array.at<float>(i, 0) = -1;
		}
		break;
	case LESS_THAN:
		for (int i = 0; i < dataMatrix.rows; i++)
		{
			float val = dataMatrix.at<float>(i, dim);
			if (val <= threshold)
				ret_array.at<float>(i, 0) = -1;
		}
		break;
	}
	*out = ret_array;
}

void AdaBoost::BuildStump(Stump* oStump, double* oErr, Mat* oBest, const Mat data_matrix, const Mat class_labels, const Mat weigh_arr)
{
	int m = data_matrix.rows;
	int n = data_matrix.cols;
	Stump best_stump = Stump();
	Mat best_class_est = Mat(m, 1, CV_32F, 0.);
	float min_error = INFINITY;

	clock_t begin, end;
	for (int i = 0; i < n; i++)
	{
		begin = clock();
		double range_min, range_max;
		minMaxLoc(data_matrix.col(i), &range_min, &range_max);
		double step_size = (range_max - range_min) / prop->stepsPerIteration;
		double weighted_error;
		Mat predicted_vals;
		Inequality inequal;
		for (int j = -1; j < (int)prop->stepsPerIteration + 1; j++)
		{
			float thresh_val = (float)(range_min + float(j) * step_size);

			for (int item = LESS_THAN; item <= BIGGER_THAN; item++)
			{
				inequal = static_cast<Inequality>(item);
				StumpClassify(&predicted_vals, data_matrix, i, thresh_val, inequal);

				//Mat err_arr = Mat(m, 1, CV_8U, 1);
				bool* err_arr = new bool[m];

				for (int index_i = 0; index_i < m; index_i++)
				{
					float pv = predicted_vals.at<float>(index_i, 0);
					float lm = class_labels.at<float>(index_i, 0);
					err_arr[index_i] = 1;
					if (pv == lm)
						err_arr[index_i] = 0;
				}

				weighted_error = inner_product((float*)weigh_arr.data, (float*)weigh_arr.data + m, err_arr, 0.0f);
				free(err_arr);
				if (weighted_error < min_error)
				{
					min_error = (float)weighted_error;
					best_class_est = Mat(predicted_vals);
					best_stump.dim = i;
					best_stump.thresh = thresh_val;
					best_stump.ineq = inequal;
				}
			}
		}
		end = clock();
	}

	*oStump = best_stump;
	*oErr = min_error;
	*oBest = best_class_est;
}

void AdaBoost::AdaboostTrainDS(vector<Stump>* weak_class_arr, Mat* agg_class_est, const Mat data_arr, const Mat class_labels)
{
	int m = data_arr.rows;
	Mat weigh_arr = 0.01 * Mat(m, 1, CV_32F, 1) / m;
	Mat out_agg_class_est = Mat(m, 1, CV_32F, Scalar(0));

	vector<Stump> out_weak_class_arr;


	for (int i = 0; i < prop->nIterations; i++)
	{
		Stump best_stump;
		double error;
		Mat class_est;
		clock_t begin, end;
		begin = clock();
		BuildStump(&best_stump, &error, &class_est, data_arr, class_labels, weigh_arr);
		end = clock();
		if (Utils::verbose)
			cout << "BuildStump..." << double(end - begin) / CLOCKS_PER_SEC << "sec" << endl;
		float alpha = float(0.5*log((1.0 - error) / max(error, 1e-16)));
		best_stump.alpha = alpha;
		out_weak_class_arr.push_back(best_stump);

		Mat expon = (-1 * alpha*class_labels).mul(class_est);
		Mat exp_expon;
		exp(expon, exp_expon);
		weigh_arr = weigh_arr.mul(exp_expon);
		weigh_arr = weigh_arr / sum(weigh_arr)[0];
		Mat prod = alpha*class_est;
		out_agg_class_est += prod;
		Mat agg_errors = Mat(out_agg_class_est.size(), CV_32F, Scalar(0));

		for (int index_i = 0; index_i < m; index_i++)
		{
			for (int index_j = 0; index_j < 1; index_j++)
			{
				float agg_element = out_agg_class_est.at<float>(index_i, index_j);
				float c_label_element = class_labels.at<float>(index_i, index_j);

				if ((agg_element > 0.0 && c_label_element < 0.0) || (agg_element < 0.0 && c_label_element > 0.0))
					agg_errors.at<float>(index_i, index_j) = 1.0;
			}
		}

		int acc_errors = (int)sum(agg_errors)[0];

		float error_rate = acc_errors / (float)m;
		if (Utils::verbose)
		{
			cout << acc_errors << " errors over " << m << " inputs..." << endl;
			cout << "Iteration {" << i << "} ---- total error: {" << error_rate << "}" << endl;
		}

		if (error_rate == 0.0)
			break;
	}
	*weak_class_arr = out_weak_class_arr;
	*agg_class_est = out_agg_class_est;
}

void AdaBoost::AdaboostTestDS(Mat* bin_class_est, Mat* agg_class_est, const Mat data_matrix, vector<Stump>* classifier_arr)
{
	int m = data_matrix.rows;
	Mat out_agg_class_est = Mat(m, 1, CV_32F, Scalar(0));
	Mat out_bin_class_est = Mat(m, 1, CV_32F, Scalar(0));

	for (unsigned int i = 0; i < classifier_arr->size(); i++)
	{
		Mat class_est;
		StumpClassify(&class_est, data_matrix, (*classifier_arr)[i].dim,
			(*classifier_arr)[i].thresh, (*classifier_arr)[i].ineq);
		out_agg_class_est += (*classifier_arr)[i].alpha * class_est;
		//add(out_agg_class_est, classifier_arr[i].alpha * class_est, out_agg_class_est);
	}


	for (int i = 0; i < m; i++)
	{
		switch (prop->ineq)
		{
		case BIGGER_THAN:
			if (out_agg_class_est.at<float>(i, 0) > prop->thresh)
				out_bin_class_est.at<float>(i, 0) = -1.f;
			if (out_agg_class_est.at<float>(i, 0) <= prop->thresh)
				out_bin_class_est.at<float>(i, 0) = 1.f;
			break;
		case LESS_THAN:
			if (out_agg_class_est.at<float>(i, 0) <= prop->thresh)
				out_bin_class_est.at<float>(i, 0) = -1.f;
			if (out_agg_class_est.at<float>(i, 0) > prop->thresh)
				out_bin_class_est.at<float>(i, 0) = 1.f;
			break;
		default:
			break;
		}
	}

	*bin_class_est = out_bin_class_est;
	*agg_class_est = out_agg_class_est;
}

void AdaBoost::WriteToFile(const vector<Stump>* stump, const char* filePath)
{
	ofstream output_file(filePath);
	
	ostream_iterator<Stump> output_iterator(output_file, "\n");
	copy(stump->begin(), stump->end(), output_iterator);
	output_file.close();
}

void AdaBoost::ReadFromFile(vector<Stump>* stump, const char* filePath)
{
	ifstream input_file(filePath);
	istream_iterator<Stump> start(input_file), end;
	vector<Stump> numbers(start, end);
	*stump = numbers;
}
