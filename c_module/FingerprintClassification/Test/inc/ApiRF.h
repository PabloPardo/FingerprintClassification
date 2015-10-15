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

struct Handle
{
	void* rTrees;
	void* norMat;
};

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType Extraction(const char*, const char*, const char*);
	__declspec(dllimport) ReturnType ExtractFingerPrint(int*, float**, unsigned char*, int w, int h, float*);
	__declspec(dllimport) ReturnType Fit(TrainPaths, const char*);
	__declspec(dllimport) ReturnType PredictTest(PredictPaths, const char*);
	__declspec(dllimport) ReturnType Predict(int*, float**, Handle*, float*);
	__declspec(dllimport) ReturnType InitModel(Handle**, const char*);
	__declspec(dllimport) ReturnType ReleaseModel(Handle*);
	__declspec(dllimport) ReturnType ReleaseFloatArrayPointer(float*);
#ifdef __cplusplus
}
#endif

#endif