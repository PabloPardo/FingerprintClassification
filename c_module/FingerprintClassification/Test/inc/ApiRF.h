#ifndef APIRF_H
#define	APIRF_H

struct ReturnType{
	int         code;
	const char* message;
};
struct TrainPaths
{
	char* labelsPath;
	char* dataPath;
};

struct PredictPaths
{
	char* labelsPath;
	char* dataPath;
};

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType Extraction(const char* labelsPath, const char* imgPath, const char* modelPath);
	__declspec(dllimport) ReturnType NormalizeFitAndPredict(TrainPaths tPaths, PredictPaths pPaths, const char* results);
#ifdef __cplusplus
}
#endif

#endif