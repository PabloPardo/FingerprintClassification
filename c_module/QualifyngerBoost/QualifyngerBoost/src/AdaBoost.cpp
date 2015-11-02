#include "AdaBoost.h"


AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}

void AdaBoost::StumpClassify(Mat* out, const Mat dataMatrix, int dim, float threshold, Inequality ineq)
{
	Mat ret_array = Mat(dataMatrix.rows, 1, CV_32F, 1.);
	
	for (int i = 0; i < dataMatrix.rows; i++)
	{
		float val = dataMatrix.at<float>(i, dim);
		switch (ineq)
		{
			case BIGGER_THAN:
				if (val > threshold)
					ret_array.at<float>(i, dim) = -1;
				break;
			case LESS_THAN:
				if (val <= threshold)
					ret_array.at<float>(i, dim) = -1;
			default:
				break;
		}
	}
	*out = ret_array;
}

void AdaBoost::BuildStump(Stump* oStump, double* oErr, Mat* oBest, const Mat data_matrix, const Mat class_labels, const Mat weigh_arr, float num_steps)
{
	Mat label_mat = class_labels.t();
	int m = data_matrix.rows;
	int n = data_matrix.cols;
	Stump best_stump = Stump();
	Mat best_class_est = Mat(m, 1, CV_8U, 0);
	float min_error = INFINITY;
	Mat ones = Mat(m, 1, CV_8U, 1);

	for (int i = 0; i < n; i++)
	{
		double range_min, range_max;
		minMaxLoc(data_matrix.col(i), &range_min, &range_max);


		double step_size = (range_max - range_min) / num_steps;
		double weighted_error;
		Mat predicted_vals;
		Inequality inequal;
		for (int j = -1; j < (int)num_steps + 1; j++)
		{
			float thresh_val = (range_min + float(j) * step_size);

			for (int item = BIGGER_THAN; item != LESS_THAN; item++)
			{
				inequal = static_cast<Inequality>(item);


				StumpClassify(&predicted_vals, data_matrix, i, thresh_val, inequal);
				Mat err_arr = Mat(ones);
				for (int index_i = 0; index_i < err_arr.rows; index_i++)
				{
					for (int index_j = 0; index_j < err_arr.cols; index_j++)
					{
						float pv = predicted_vals.at<float>(index_i, index_j);
						int lm = label_mat.at<int>(index_i, index_j);
						if ((int)pv == lm)
							err_arr.at<int>(index_i, index_j) = 0;
						else if (pv == 1 && lm == -1)
							err_arr.at<int>(index_i, index_j) = 3;
					}
				}
				weighted_error = weigh_arr.t().dot(err_arr);
			}

			if (weighted_error < min_error)
			{
				min_error = weighted_error;
				best_class_est = Mat(predicted_vals);
				best_stump.dim = i;
				best_stump.thresh = thresh_val;
				best_stump.ineq = inequal;
			}
		}
	}
	
	*oStump = best_stump;
	*oErr = min_error;
	*oBest = best_class_est;
}

void AdaBoost::AdaboostTrainDS(vector<Stump>* weak_class_arr, Mat* agg_class_est, const Mat data_arr, const Mat class_labels, int num_it, int num_ds_steps)
{
	int m = data_arr.rows;
	Mat weigh_arr = 0.01 * Mat(m, 1, CV_32F, 1) / m;
	Mat out_agg_class_est = Mat(m, 1, CV_32F, 0);
	vector<Stump> out_weak_class_arr;

	for (int i = 0; i < num_it; i++)
	{
		Stump best_stump;
		double error;
		Mat class_est;
		BuildStump(&best_stump, &error, &class_est, data_arr, class_labels, weigh_arr, num_ds_steps);

		float alpha = float(0.5*log((1.0 - error) / max(error, 1e-16)));
		best_stump.alpha = alpha;
		out_weak_class_arr.push_back(best_stump);
		Mat class_labels_t = class_labels.t();
		Mat expon = (-1*alpha*class_labels_t).mul(class_est);
		Mat exp_expon;
		exp(expon, exp_expon);
		weigh_arr = weigh_arr.mul(exp_expon);
		weigh_arr = weigh_arr / sum(weigh_arr)[0];
		add(out_agg_class_est, alpha*class_est, out_agg_class_est);
		Mat agg_errors = Mat(data_arr.size(),CV_8U);

		for (int index_i = 0; index_i < m; i++)
		{
			for (int index_j = 0; index_j < 1; index_j++)
			{
				agg_errors.at<uchar>(index_i, index_j) = out_agg_class_est.at<float>(index_i, index_j) > 0 && class_labels_t.at<uchar>(index_i, index_j) < 0;
			}
		}

		float error_rate = sum(agg_errors)[0] / m;

		if (error_rate == 0.0)
			break;
	}
	*weak_class_arr = out_weak_class_arr;
	*agg_class_est = out_agg_class_est;
}

void AdaBoost::AdaboostTestDS(Mat* bin_class_est, Mat* agg_class_est, const Mat data_matrix, const vector<Stump> classifier_arr, int thresh, Inequality ineq)
{
	int m = data_matrix.row;
	Mat out_agg_class_est = Mat(m, 1, CV_8U);
	Mat out_bin_class_est = Mat(m, 1, CV_8U);
	
	for (int i = 0; i < classifier_arr.size(); i++)
	{
		Mat class_est;
		StumpClassify(&class_est, data_matrix, classifier_arr[i].dim, 
			classifier_arr[i].thresh, classifier_arr[i].ineq);
		add(out_agg_class_est, classifier_arr[i].alpha * class_est, out_agg_class_est);
	}
	

	for (int i = 0; i < m; i++)
	{
		switch (ineq)
		{
		case BIGGER_THAN:
			if (out_agg_class_est.at<uchar>(i, 0) > thresh)
				out_bin_class_est.at<uchar>(i, 0) = -1;
			if (out_agg_class_est.at<uchar>(i, 0) <= thresh)
				out_bin_class_est.at<uchar>(i, 0) = 1;
			break;
		case LESS_THAN:
			if (out_agg_class_est.at<uchar>(i, 0) <= thresh)
				out_bin_class_est.at<uchar>(i, 0) = -1;
			if (out_agg_class_est.at<uchar>(i, 0) > thresh)
				out_bin_class_est.at<uchar>(i, 0) = 1;
			break;
		default:
			break;
		}
	}
	
	*bin_class_est = out_bin_class_est;
	*agg_class_est = out_agg_class_est;
}
