#include <iostream>
#include <opencv2\ml\ml.hpp>

#ifndef FINGER_PRINT_CLASSIFICATION_H
#define FINGER_PRINT_CLASSIFICATION_H

struct ReturnType{
	int         code;
	const char* message;
};

struct Properties
{
	int n_bins; //	Number of histogram bins.
	int rad_grad; // Gradient Radius.
	int rad_dens; // Density Radius.
	int rad_entr; // Entropy Radius.
	int max_depth; // Max depth of the trees in the Random Forest.
	int min_samples_count; // Min samples needed to split a leaf.
	int max_categories; // Max number of categories.
	int max_num_of_trees_in_forest; // Max number of trees in the forest.
	int nactive_vars; // nactive_vars,

	bool verbose;

	Properties() {  
		n_bins = 32; 
		rad_grad=1; 
		rad_dens=3; 
		rad_entr=5; 
		max_depth=16; 
		min_samples_count=2; 
		max_categories=3; 
		max_num_of_trees_in_forest=10;
		nactive_vars = 0;
		verbose = false;
	};
	friend std::ostream& operator<<(std::ostream& os, const Properties& prop);
};

struct PropertiesSVM
{
	int n_bins; //	Number of histogram bins.
	int rad_grad; // Gradient Radius.
	int rad_dens; // Density Radius.
	int rad_entr; // Entropy Radius.
	int svm_type; // Type of SVM.
	int kernel_type; //Type of kernel.
	double degree; //Parameter degree of a kernel function (POLY).
	double gamma; // Parameter \gamma of a kernel function (POLY / RBF / SIGMOID).
	double coef0; // Parameter coef0 of a kernel function (POLY / SIGMOID).
	double Cvalue; // Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
	double nu; // Parameter \nu of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
	double p; // Parameter \epsilon of a SVM optimization problem (EPS_SVR).
	int weights;
	CvTermCriteria term_crit;

	bool verbose;

	PropertiesSVM() {  
		n_bins = 32; 
		rad_grad=1; 
		rad_dens=3; 
		rad_entr=5; 
		svm_type=CvSVM::C_SVC; 
		kernel_type=CvSVM::RBF; 
		degree=0;
		gamma=1;
		coef0=0;
		Cvalue=1;
		nu=0;
		p=0;
		weights=0;
		term_crit=cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
		verbose = false;
	};
	friend std::ostream& operator<<(std::ostream& os, const PropertiesSVM& propSVM);
};

extern "C" __declspec(dllexport) ReturnType ReleaseFloatPointer(float*);

extern "C" __declspec(dllexport) ReturnType InitModel(CvRTrees**,const char*);

extern "C" __declspec(dllexport) ReturnType InitModelSVM(CvSVM**,const char*);

extern "C" __declspec(dllexport) ReturnType ReleaseModel(CvRTrees*);

extern "C" __declspec(dllexport) ReturnType ReleaseModelSVM(CvSVM*);

extern "C" __declspec(dllexport) ReturnType SetProperties(Properties*);

extern "C" __declspec(dllexport) ReturnType SetPropertiesSVM(PropertiesSVM*);

/**************************************************************************
*								  FitRF
*								  -----
*		imagesDir	: Path to the training images.
*		paramsFeat  : List of parameters needed to extract features.
*						- Number of histogram bins
*						- Gradient Radius
*						- Density Radius
*						- Entropy Radius
*		paramsRF	: List of parameters needed to train.
*						- max_depth
*						- min_sample_count
*						- max_categories
*						- max_num_of_trees_in_the_forest
*		outPath		: Path where the trained model will be saved.
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType FitRF(char*, char*, char*);

extern "C" __declspec(dllexport) ReturnType FitFromDataRF(char*, char*, char*, bool);

/**************************************************************************
*								  FitSVM
*								  ------
*		imagesDir	: Path to the training images.
*		paramsFeat  : List of parameters needed to extract features.
*						- Number of histogram bins
*						- Gradient Radius
*						- Density Radius
*						- Entropy Radius
*		paramsRF	: List of parameters needed to train.
*						- max_depth
*						- min_sample_count
*						- max_categories
*						- max_num_of_trees_in_the_forest
*		outPath		: Path where the trained model will be saved.
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType FitSVM(char*, char*, char*);

extern "C" __declspec(dllexport) ReturnType FitFromDataSVM(char*, char*, char*, bool);

/**************************************************************************
*								PredictRF
*								---------
*		imagePath  : Path to the image we want to predict on.
*		modelPath  : Path to the trained model file.
*		paramsFeat : List of parameters needed to extract features.
*						- Number of histogram bins
*						- Gradient Radius
*						- Density Radius
*						- Entropy Radius
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType PredictRF(float**, unsigned char*, int, int, const char*, void*, const float*);

/**************************************************************************
*								PredictSVM
*								----------
*		imagePath  : Path to the image we want to predict on.
*		modelPath  : Path to the trained model file.
*		paramsFeat : List of parameters needed to extract features.
*						- Number of histogram bins
*						- Gradient Radius
*						- Density Radius
*						- Entropy Radius
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType PredictSVM(int**, unsigned char*, int, int, const char*, void*, const float*);

/**************************************************************************
*								CrossPredictRF
*								--------------
*		modelPath			: Path to the trained model file.
*		normalizedFeatures	: Normalized Features from an image.
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType PredictFromDataRF(float**, void*, double*);

/**************************************************************************
*								CrossPredictSVM
*								---------------
*		modelPath			: Path to the trained model file.
*		normalizedFeatures	: Normalized Features from an image.
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType PredictFromDataSVM(int**, void*, double*);

/**************************************************************************
*								ExtractFeatures
*								---------
*		csvPath				: Path to the file with the fingerPrint features.
*		imagesPath			: Path to the fingerprint image collection.
*		outPath				: Output path with the results {normalization, unnormalizedData, normalizedData}
*		prefix				: Prefix of output file
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType ExtractFeatures(char*,char*,char*,char*);

/**************************************************************************
*								ExportMeanStdFile
*								---------
*		unNormalizedDataPath	: Path to the file with the fingerPrint features.
*		outPath					: Output path with the results {normalization, unnormalizedData, normalizedData}
*		verbose					
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType ExportMeanStdFile(const char*, const char*, bool verbose=false);


/**************************************************************************
*								PredictFromLabelsAndFeatureFile
*								---------
*		labelsPath				: Path to the file with the labels of each image.
*		imagesPath				: Path to the fingerprint grayscale image
*		modelPath				: Path to the trained model					
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType PredictFromLabelsAndFeatureFile(const char*, const char*, const char*);

#endif