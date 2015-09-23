#include <iostream>

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

extern "C" __declspec(dllexport) ReturnType ReleaseFloatPointer(float*);

extern "C" __declspec(dllexport) ReturnType InitModel(void**,const char*);

extern "C" __declspec(dllexport) ReturnType ReleaseModel(void*);

extern "C" __declspec(dllexport) ReturnType SetProperties(Properties*);

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
*								CrossPredictRF
*								---------
*		modelPath			: Path to the trained model file.
*		normalizedFeatures	: Normalized Features from an image.
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType CrossPredictRF(float**, void*, double*);

/**************************************************************************
*								ExtractFeatures
*								---------
*		csvPath				: Path to the file with the fingerPrint features.
*		imagesPath			: Path to the fingerprint image collection.
*		outPath				: Output path with the results {normalization, unnormalizedData, normalizedData}
***************************************************************************/
extern "C" __declspec(dllexport) ReturnType ExtractFeatures(char*,char*,char*);

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