#ifndef APIRF_H
#define	APIRF_H

struct ReturnType{
	int         code;
	const char* message;
};

struct MatRange
{
	int begin;
	int end;
};

struct LoadCsvParams
{
	bool withHeaders = false;
	const char* csvFile = 0;
	const char* baseImgPath = 0;
	char separator = 0;
	MatRange globalRange;
	MatRange* XRange;
	MatRange* yRange;
	int fileNameIndex = -1;
	int folderNameIndex = -1;
	const char* fingerFieldName = 0;;
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

struct ImgProcessingProperties
{
	int n_bins; //	Number of histogram bins.
	int rad_grad; // Gradient Radius.
	int rad_dens; // Density Radius.
	int rad_entr; // Entropy Radius.
};
enum Inequality
{
	LESS_THAN,
	BIGGER_THAN
};
struct AdaBoostProperties
{
	int nIterations;
	int stepsPerIteration;

	float thresh;
	Inequality ineq;
};

struct RFProperties
{
	int max_depth; // Max depth of the trees in the Random Forest.
	int min_samples_count; // Min samples needed to split a leaf.
	int max_categories; // Max number of categories.
	int max_num_of_trees_in_forest; // Max number of trees in the forest.
	int nactive_vars; // nactive_vars,
};

struct Config
{
	bool verbose;
	ImgProcessingProperties* extractionProperties;
	AdaBoostProperties* adaBoostProperties;
	RFProperties* randomForestProperties;
};

#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllimport) ReturnType InitConfig(Config);
	__declspec(dllimport) ReturnType Extraction(LoadCsvParams, const char*);
	__declspec(dllimport) ReturnType ExtractFingerPrint(int*, float**, unsigned char*, int w, int h, float*);
	__declspec(dllimport) ReturnType Fit1(LoadCsvParams, const char*, const char*);
	__declspec(dllimport) ReturnType Fit2(LoadCsvParams, const char*, const char*);
	__declspec(dllimport) ReturnType Normalize(const char*, const char*, const char*);
	__declspec(dllimport) ReturnType Predict1(bool*, Handle*, float*);
	__declspec(dllimport) ReturnType Predict2(int*, float**, Handle*, float*);
	__declspec(dllimport) ReturnType InitModel(Handle**, const char*);
	__declspec(dllimport) ReturnType ReleaseModel(Handle*);
	__declspec(dllimport) ReturnType ReleaseFloatArrayPointer(float*);
	__declspec(dllimport) ReturnType PredictTest(LoadCsvParams, const char*, const char*, const char*);
#ifdef __cplusplus
}
#endif

#endif