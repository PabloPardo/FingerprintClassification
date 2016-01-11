#include "AdaBoost.h"
#include <vector>

struct ReturnType{
	int         code;
	const char* message;
};

AdaBoost* obj = new AdaBoost();

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) ReturnType SetVerbose(bool verbose)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			AdaBoost::verbose = verbose;
		}
		catch (cv::Exception& ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception& ex)
		{
			ret.code = 2;
			ret.message = ex.what();
		}
		catch (...)
		{
			ret.code = 3;
			ret.message = "Unknown error";
		}
		return ret;
	}

	

	__declspec(dllexport) ReturnType Train(vector<Stump>* weak_class_arr, Mat* agg_class_est, const Mat data_arr, const Mat class_labels, int num_it, int num_ds_steps)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			
			obj->AdaboostTrainDS(weak_class_arr, agg_class_est, data_arr, class_labels, num_it, num_ds_steps);

		}
		catch (cv::Exception& ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception& ex)
		{
			ret.code = 2;
			ret.message = ex.what();
		}
		catch (...)
		{
			ret.code = 3;
			ret.message = "Unknown error";
		}
		return ret;
	}

	__declspec(dllexport) ReturnType Test(Mat* bin_class_est, Mat* agg_class_est, const Mat data_matrix, vector<Stump>* classifier_arr, float thresh, Inequality ineq)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->AdaboostTestDS(bin_class_est, agg_class_est, data_matrix, classifier_arr, thresh, ineq);
		}
		catch (cv::Exception& ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception& ex)
		{
			ret.code = 2;
			ret.message = ex.what();
		}
		catch (...)
		{
			ret.code = 3;
			ret.message = "Unknown error";
		}
		return ret;
	}

	__declspec(dllexport) void WriteToFile(vector<Stump>* param, const char* filePath)
	{
		obj->WriteToFile(param, filePath);
	}

	__declspec(dllexport) void ReadFromFile(vector<Stump>* param, const char* filePath)
	{
		obj->ReadFromFile(param, filePath);
	}

#ifdef __cplusplus
}
#endif