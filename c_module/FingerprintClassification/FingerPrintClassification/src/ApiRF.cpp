#include "utils.h"
#include "MainRF.h"

MainRF* obj = new MainRF();
extern "C" __declspec(dllexport) ReturnType ExtractNormalizeAndFit(const char* labelsPath, const char* imgPath, const char* modelPath)
{
	ReturnType ret = { 0, "No Error" };
	try
	{
		obj->ExtractNormalizeAndFit(labelsPath, imgPath, modelPath);
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