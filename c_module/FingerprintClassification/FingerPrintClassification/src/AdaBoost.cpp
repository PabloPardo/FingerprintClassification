#include "AdaBoost.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <numeric>
#include "utils.h"
#include <iterator>


AdaBoost::AdaBoost()
{
}

AdaBoost::~AdaBoost()
{
}

using namespace std;

bool AdaBoost::verbose;

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

void AdaBoost::BuildStump(Stump* oStump, double* oErr, Mat* oBest, const Mat data_matrix, const Mat class_labels, const Mat weigh_arr, int num_steps)
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
		double step_size = (range_max - range_min) / num_steps;
		double weighted_error;
		Mat predicted_vals;
		Inequality inequal;
		for (int j = -1; j < (int)num_steps + 1; j++)
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

void AdaBoost::AdaboostTrainDS(vector<Stump>* weak_class_arr, Mat* agg_class_est, const Mat data_arr, const Mat class_labels, int num_it, int num_ds_steps)
{
	int m = data_arr.rows;
	Mat weigh_arr = 0.01 * Mat(m, 1, CV_32F, 1) / m;
	Mat out_agg_class_est = Mat(m, 1, CV_32F, Scalar(0));

	vector<Stump> out_weak_class_arr;


	for (int i = 0; i < num_it; i++)
	{
		Stump best_stump;
		double error;
		Mat class_est;
		clock_t begin, end;
		begin = clock();
		BuildStump(&best_stump, &error, &class_est, data_arr, class_labels, weigh_arr, num_ds_steps);
		end = clock();
		if (AdaBoost::verbose)
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
		if (AdaBoost::verbose)
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

void AdaBoost::AdaboostTestDS(Mat* bin_class_est, Mat* agg_class_est, const Mat data_matrix, vector<Stump>* classifier_arr, float thresh, Inequality ineq)
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
		switch (ineq)
		{
		case BIGGER_THAN:
			if (out_agg_class_est.at<float>(i, 0) > thresh)
				out_bin_class_est.at<float>(i, 0) = -1.f;
			if (out_agg_class_est.at<float>(i, 0) <= thresh)
				out_bin_class_est.at<float>(i, 0) = 1.f;
			break;
		case LESS_THAN:
			if (out_agg_class_est.at<float>(i, 0) <= thresh)
				out_bin_class_est.at<float>(i, 0) = -1.f;
			if (out_agg_class_est.at<float>(i, 0) > thresh)
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

int AdaBoost::LoadCSV(CsvData* out, LoadCsvParams params)
{
	CsvData tmp;
	int nLines;

	if (verbose)
		cout << "Counting file lines... " << endl;
	nLines = countLines(params.csvFile);
	if (verbose)
		cout << "Counting file lines completed: " << nLines << endl;

	ifstream ifs(params.csvFile, ifstream::in);

	if (!ifs.is_open())
		return -1;
	int cont;
	string line;
	string value;
	istringstream iss;
	if (params.with_headers)
	{
		tmp.body = Mat(nLines - 1, params.end_header - params.begin_header, CV_32F);
		getline(ifs, line);
		iss = istringstream(line);

		cont = 0;

		while (cont < params.begin_header)
		{
			getline(iss, value, params.separator);
			cont++;
		}

		for (int j = params.begin_header; j < params.end_header; j++)
		{
			getline(iss, value, params.separator);
			tmp.headers.push_back(value);
		}
	}
	else
		tmp.body = Mat(nLines, params.end_header - params.begin_header, CV_32F);

	for (int i = 0; i < tmp.body.rows; i++)
	{
		getline(ifs, line);
		iss = istringstream(line);
		cont = 0;
		while (cont < params.begin_header)
		{
			getline(iss, value, params.separator);
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

int AdaBoost::DigitalizeFingers(Mat* output, const Mat onlyFingerNumber)
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
