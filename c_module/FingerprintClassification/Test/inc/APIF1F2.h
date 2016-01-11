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

struct LoadCsvParams
{
	bool with_headers;
	char* csvFile;
	char separator;
	int begin_header;
	int end_header;
};

struct PredictPaths
{
	const char* labelsPath;
	const char* dataPath;
	const char* modelDir;
};

struct Handle
{
	/*Fase 1*/
	void* norFase1;
	void* vec_stumps;
	/*Fase 2*/
	void* norMat;
	void* rTrees;
};

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType Extraction(const char*, const char*, const char*);
	__declspec(dllimport) ReturnType ExtractFingerPrint(int*, float**, unsigned char*, int w, int h, float*);
	__declspec(dllimport) ReturnType Fit1(LoadCsvParams, const char*, int, int);
	__declspec(dllimport) ReturnType Fit2(TrainPaths, const char*);
	__declspec(dllimport) ReturnType Normalize(const char*, const char*, const char*);
	__declspec(dllimport) ReturnType PredictTest(PredictPaths, const char*);
	__declspec(dllimport) ReturnType Predict1(bool*, Handle*, float, float*);
	__declspec(dllimport) ReturnType Predict2(int*, float**, Handle*, float*);
	__declspec(dllimport) ReturnType InitModel(Handle**, const char*);
	__declspec(dllimport) ReturnType ReleaseModel(Handle*);
	__declspec(dllimport) ReturnType ReleaseFloatArrayPointer(float*);
#ifdef __cplusplus
}
#endif

#endif