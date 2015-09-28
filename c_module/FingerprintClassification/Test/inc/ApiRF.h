#ifndef APIRF_H
#define	APIRF_H

struct ReturnType{
	int         code;
	const char* message;
};
struct TrainPaths
{
	const char* labelsPath;
	const char* dataPath;
};

struct PredictPaths
{
	const char* labelsPath;
	const char* dataPath;
	const char* modelDir;
};

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType Extraction(const char*, const char*, const char*);
	__declspec(dllimport) ReturnType ExtractFingerPrint(float**, unsigned char*, int w, int h, float*);
	__declspec(dllimport) ReturnType Fit(TrainPaths, const char*);
	__declspec(dllimport) ReturnType PredictTest(PredictPaths, const char*);
	__declspec(dllimport) ReturnType Predict(float**, PredictPaths, float*);
#ifdef __cplusplus
}
#endif

#endif