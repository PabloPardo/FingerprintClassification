#include "utils.h"
#include "MainFP.h"

struct ReturnType{
	int         code;
	const char* message;
};

MainFP* obj = new MainFP();

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) ReturnType InitConfig(Config cfg)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->InitConfig(cfg);
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

	__declspec(dllexport) ReturnType Extraction(LoadCsvParams params, const char* modelPath)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Extraction(params, modelPath);
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

	__declspec(dllexport) ReturnType Normalize(const char* inputDir, const char* inputFile, const char* outDir)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Normalize(inputDir, inputFile, outDir);
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

	__declspec(dllexport) ReturnType Fit1(LoadCsvParams importData, const char* extractedFileFase1, const char* outputPath)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Fit1(importData, extractedFileFase1, outputPath);
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

	__declspec(dllexport) ReturnType Fit2(LoadCsvParams params, const char* dataPath, const char* outputDir)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Fit2(params, dataPath, outputDir);
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

	__declspec(dllexport) ReturnType Predict1(bool* enoughQuality, Handle* hnd, float* features)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->Predict1(enoughQuality,  hnd, features);
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
	
	__declspec(dllexport) ReturnType Predict2(int* lenProbs, float** probs, Handle* hnd, float* features)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			/*cout << "rTrees:" << ptrModel << endl;
			cout << "norMat:" << ptrNorMat << endl;*/
			obj->Predict2(lenProbs, probs, hnd, features);
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

	__declspec(dllexport) ReturnType PredictTest(LoadCsvParams params, const char* modelDir, const char* dataPath, const char* results)
	{
		ReturnType ret = { 0, "No Error" };
		try
		{
			obj->PredictTest(params, modelDir, dataPath, results);
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

#ifdef __cplusplus
}
#endif