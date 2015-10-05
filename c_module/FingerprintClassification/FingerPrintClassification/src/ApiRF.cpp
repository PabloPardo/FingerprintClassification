#include "utils.h"
#include "MainRF.h"

struct ReturnType{
	int         code;
	const char* message;
};

MainRF* obj = new MainRF();

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) ReturnType Extraction(const char* labelsPath, const char* imagesPath, const char* outPath)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Extraction(labelsPath, imagesPath, outPath);
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

	__declspec(dllexport) ReturnType Fit(const TrainPaths tPaths, const char* outputDir)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Fit(tPaths, outputDir);
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

	__declspec(dllexport) ReturnType PredictTest(PredictPaths pPaths, const char* results)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->PredictTest(pPaths, results);
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

	__declspec(dllexport) ReturnType Predict(int* lenProbs, float** probs, Handle* hnd, float* features)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			/*cout << "rTrees:" << ptrModel << endl;
			cout << "norMat:" << ptrNorMat << endl;*/
			obj->Predict(lenProbs, probs, hnd, features);
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

	__declspec(dllexport) ReturnType ExtractFingerPrint(int* lenFeatures, float** features, unsigned char* img, int w, int h, float* nfiq)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->ExtractFingerPrint(lenFeatures, features, img, w, h, nfiq);
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

	__declspec(dllexport) ReturnType InitModel(Handle** handle, const char *modelPath)
	{
		ReturnType ret = { 0, "No Error" };

		try
		{
			obj->InitModel(handle, modelPath);
		}
		catch (cv::Exception ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception ex)
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

	__declspec(dllexport) ReturnType ReleaseModel(Handle* hnd)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->ReleaseModel(hnd);
		}
		catch (cv::Exception ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception ex)
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

	__declspec(dllexport) ReturnType ReleaseFloatArrayPointer(float* arr)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->ReleaseFloatPointer(arr);
		}
		catch (cv::Exception ex)
		{
			ret.code = 1;
			ret.message = ex.what();
		}
		catch (std::exception ex)
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

#ifdef __cplusplus
}
#endif