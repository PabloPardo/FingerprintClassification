#include "utils.h"
#include "MainRF.h"

MainRF* obj = new MainRF();
extern "C" __declspec(dllexport) ReturnType Extraction(const char* labelsPath, const char* imagesPath,const char* outPath)
{
	ReturnType ret = { 0, "No Error" };
	try
	{
		obj->Extraction(labelsPath, imagesPath, outPath);
	}
	catch (std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	catch (cv::Exception& ex)
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

extern "C" __declspec(dllexport) ReturnType NormalizeFitAndPredict(TrainPaths tPaths, PredictPaths pPaths, const char* results)
{
	ReturnType ret = { 0, "No Error" };
	obj->NormalizeFitAndPredict(tPaths, pPaths, results);
	return ret;
}