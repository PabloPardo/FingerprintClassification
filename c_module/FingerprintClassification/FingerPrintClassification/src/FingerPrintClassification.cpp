#include <FingerPrintClassification.h>
#include <FpFeaturesLibrary.h>
#include <utils.h>
#include <cctype>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <Windows.h>
#include <io.h>


Properties* prop = new Properties();
Config cfg = Config();
/**************************************************************************
*								  Segment
*								  -------
*		img		 : Image.
*		bin_thr  : Threshold to binarize the image.
*
***************************************************************************/
cv::Mat Segmenta(cv::Mat img, int bin_thr = 250){

	// Binarize image
    cv::Mat binaryMat(img.size(), img.type());
    cv::threshold(img, binaryMat, bin_thr, 255, CV_THRESH_BINARY_INV);

	// Axis Projections
	cv::Mat rows, cols;
	cv::reduce(binaryMat, rows, 1, CV_REDUCE_SUM, CV_32S);
	cv::reduce(binaryMat, cols, 0, CV_REDUCE_SUM, CV_32S);

	// Select cropping rectangel
	int i, min_x, max_x, min_y, max_y;

	for(i=0; i<rows.rows; i++)
		if(rows.at<int>(i,0) > 0){
			min_x = i;
			break;
		}
	for(i=rows.rows-1; i>0; i--)
		if(rows.at<int>(i,0) > 0){
			max_x = i;
			break;
		}
	for(i=0; i<cols.cols; i++)
		if(cols.at<int>(0,i) > 0){
			min_y = i;
			break;
		}
	for(i=cols.cols-1; i>0; i--)
		if(cols.at<int>(0,i) > 0){
			max_y = i;
			break;
		}

	// Crop image
	cv::Rect myROI(min_y, min_x, max_y - min_y, max_x - min_x);
	cv::Mat croppedImage = img(myROI);

	/*cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", croppedImage);
    cv::waitKey(0);*/

	return croppedImage;
}


/**************************************************************************
*						CropImage
*						----------------------
*		params->
*			row			: Row selected
*			col			: Col selected
*			img			: Reference to the image.
*		returns->
*			cv::Mat subset of img region(row,col)
*
***************************************************************************/
cv::Mat CropImage(int row, int col, const cv::Mat* img) 
{
	int row_split = img->rows / Constants::NUM_ROW_SEGMENTS;
	int col_split = img->cols / Constants::NUM_COL_SEGMENTS;
	int mod_row_split = img->rows % Constants::NUM_ROW_SEGMENTS;
	int mod_col_split = img->cols % Constants::NUM_COL_SEGMENTS;
	
	int start_row = row*row_split;
	int end_row = (row + 1)*row_split;
	int start_col = col*col_split;
	int end_col = (col + 1)*col_split;
	
	if(end_row == img->rows - mod_row_split)
		end_row += mod_row_split;
	
	if(end_col == img->cols - mod_col_split)
		end_col += mod_col_split;

	cv::Rect rect = cv::Rect(start_col,start_row,end_col-start_col,end_row-start_row);

	return cv::Mat((*img)(rect));
}

/**************************************************************************
*						GetImageRegions
*						----------------------
*		params->
*			img		 : Reference to the image.
*		returns->
*			array of cv::Mat with every region
*
***************************************************************************/
cv::Mat** GetImageRegions(const cv::Mat *img) {

	cv::Mat** ret = new cv::Mat*[Constants::NUM_ROW_SEGMENTS];
	for(int i = 0; i < Constants::NUM_ROW_SEGMENTS; ++i)
		ret[i] = new cv::Mat[Constants::NUM_COL_SEGMENTS];
	
	for(int i = 0; i < Constants::NUM_ROW_SEGMENTS; i++) {
		for(int j = 0; j < Constants::NUM_COL_SEGMENTS; j++) {
			ret[i][j] = CropImage(i,j, img);
		}	
	}
	return ret;
}

/**************************************************************************
*						LoadImage_and_FeatExtr
*						----------------------
*		path	 : Path to the image.
*		n_bins   : Number of histogram bins.
*		rad_grad : Gradient Radius.
*		rad_dens : Density Radius.
*		rad_entr : Entropy Radius.
*
***************************************************************************/
cv::Mat LoadImage_and_FeatExtr (unsigned char* data, int w, int h, const cv::Mat features, int cont=0){

	// Read image
	//cv::Mat in = cv::imread(path, cv::IMREAD_GRAYSCALE); //CV_8U
	cv::Mat img = cv::Mat(w,h,CV_8U,data);
	// Segmentation
	// This call makes the histogram extraction perform slower
	//cv::Mat seg_img = Segmenta(in, 250);
	
	// Add Geyce features
	cv::Mat ret = features;
	cv::Mat dif = diferentiate_img(&img);
	//std::cout << "dif fet" << std::endl;
	cv::hconcat(dif, ret, ret);
	//std::cout << "dif concat" << std::endl;
	cv::Mat** regions = GetImageRegions(&img);
	//std::cout << "regions fetes" << std::endl;

	for(int i = Constants::NUM_ROW_SEGMENTS - 1; i >= 0; i--)
	{
		for(int j = Constants::NUM_COL_SEGMENTS - 1; j >=0 ; j--)
		{
			/*
			cfg.verboseHough = true;
			cfg.path = "out\\";
			std::ostringstream os;
			os << cfg.fileName << i << "-" << j;
			cfg.fileName = (char*)os.str().c_str();
			setConfig(&cfg);	
			*/
			cv::Mat in = regions[i][j];

			cv::Mat out_grad = hist_grad(&in, prop->rad_grad, prop->n_bins);

			cv::Mat out_dens = hist_density(&in, prop->rad_dens, prop->n_bins);
			
			cv::Mat out_hough = hist_hough(&in, prop->n_bins);

			cv::Mat out_entropy = hist_entropy(&in, prop->rad_entr, prop->n_bins);	

			// Join histograms
			cv::hconcat(out_hough, ret,  ret);
			cv::hconcat(out_entropy, ret, ret);
			cv::hconcat(out_grad, ret, ret);
			cv::hconcat(out_dens, ret, ret);
		}
	}
	
	for(int i = 0; i < Constants::NUM_ROW_SEGMENTS; ++i) {
		delete [] regions[i];
	}
	delete [] regions;
	
	cv::Mat out_grad = hist_grad(&img, prop->rad_grad, prop->n_bins);

	cv::Mat out_dens = hist_density(&img, prop->rad_dens, prop->n_bins);

	cv::Mat out_hough = hist_hough(&img, prop->n_bins);

	cv::Mat out_entropy = hist_entropy(&img, prop->rad_entr, prop->n_bins);	

	// Join histograms	
	cv::hconcat(out_hough, ret,  ret);
	cv::hconcat(out_entropy, ret, ret);
	cv::hconcat(out_grad, ret, ret);
	cv::hconcat(out_dens, ret, ret);
	if(cont > 0)
	{
		cv::Mat matCont = cv::Mat(1,1,CV_32F);
		matCont.at<float>(0,0) = cont;
		cv:hconcat(matCont, ret, ret);
	}
	return ret;
}

/**************************************************************************
*								  FitRF
*								  -----
*		csvPath						    : Path to the csv data.
*		imagesPath						: Path to the images data.
*		outPath							: Path where the trained model will be saved.
*
***************************************************************************/
ReturnType FitRF(char *csvPath, char *imagesPath, char *outPath) {

	ReturnType ret = { 0, "No Error" };
	try
	{
		/****************************************************/
		/*		     Load Image & Feature Extraction		*/
		/****************************************************/
		if(prop->verbose)
		{
			std::cout << *prop << std::endl;
		}
		//Read csv with image name + labels
		LabelsAndFeaturesData dat = readCSV(csvPath);
		//cv::Mat trainClasses = dat.matrix.t();
		std::vector<std::string> imgFileNames = dat.imgFileNames;
		cv::Mat features = dat.features;
		cv::Mat trainSamples;
		cv::Mat normTS;
		
		if (CreateDirectory(outPath, NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			for(int i = 0; i != imgFileNames.size(); i++){
				clock_t time_a = clock();
				cv::Mat featurei = features.row(i);
				cv::Mat in = cv::imread(imagesPath + imgFileNames[i],cv::IMREAD_GRAYSCALE);
				cfg.fileName = (char*)imgFileNames[i].c_str();
				if(in.rows == 0) {
					throwError("ERROR: file " + (std::string)imagesPath + imgFileNames[i] + " could not be opened. Is the path okay?");
				}
				if(prop->verbose)
					std::cout << imgFileNames[i] << " img "  << i; 
				cv::Mat hist = LoadImage_and_FeatExtr(in.data, in.rows, in.cols, featurei, 0);
				// Join features
				if(trainSamples.rows == 0)
					trainSamples = hist;
				else
					cv::vconcat(trainSamples, hist, trainSamples);
				hist.release();
				clock_t time_b = clock();
				if(prop->verbose)
					std::cout << "train. length: [" << trainSamples.rows << "," << trainSamples.cols  << "] time: " << (long)(time_b - time_a) << std::endl;
			}
			if(prop->verbose)
			{
				std::cout << *prop << std::endl;
			}
			
			normTS = createNormalizationFile(outPath,trainSamples);
			trainSamples.release();
			
			if(prop->verbose)
			{
				exportFileFeatures(normTS, imgFileNames, ((std::string)outPath + "/NormalizedData.csv").c_str());
			}
			/****************************************************/
			/*					  Train RF						*/
			/****************************************************/
			
			// Construct the classifier and set the parameters
			//CvRTrees  rtrees[Constants::NUM_CLASSIFIERS][1];
			//CvRTrees  rtrees[6][1];
			// Construct the classifier and set the parameters
			CvRTrees** rtrees;
			allocateRtrees(&rtrees,Constants::NUM_CLASSIFIERS,1);
			float priors[] = {1,1,1,1,1,1};
			CvRTParams  params( prop->max_depth,						// max_depth,
								prop->min_samples_count,				// min_sample_count,
								0.f,							// regression_accuracy,
								false,							// use_surrogates,
								prop->max_categories,					// max_categories,
								priors,								// priors,
								false,							// calc_var_importance,
								prop->nactive_vars,				// nactive_vars,
								prop->max_num_of_trees_in_forest, // max_num_of_trees_in_the_forest,
								0,								// forest_accuracy,
								CV_TERMCRIT_ITER				// termcrit_type
							   );

			// define all the attributes as numerical
			// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
			// that can be assigned on a per attribute basis
			cv::Mat var_type = cv::Mat(Constants::TOTAL_FEATURES + 1, 1, CV_8U );
			var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
			// this is a classification problem (i.e. predict a discrete number of class
			// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
			var_type.at<uchar>(Constants::TOTAL_FEATURES, 0) = CV_VAR_CATEGORICAL;

			for(int i=0; i < Constants::NUM_CLASSIFIERS; i++){

				//Prepare trainClasses for the oneVsAll classification strategy
				//cv::Mat trainClass_i = oneVsAll(trainClasses, i);
				cv::Mat trainClass_i = dat.matrix.col(i);

				// Fit the classifier with the training data
				rtrees[i]->train( normTS, CV_ROW_SAMPLE, trainClass_i, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params );

				// Save Model
				char fileName[10000];
				sprintf(fileName, "%smodel_%d.xml", outPath, i);
				rtrees[i]->save(fileName);
			}
			releaseRTrees(rtrees,Constants::NUM_CLASSIFIERS,1);
		} else {
			throw (std::string)"Folder " + outPath + " could not be created";
		}
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	catch(...)
	{
		std::cout << "General error" << std::endl;
	}
	return ret;
}

/**************************************************************************
*								  FitFromDataRF
*								  -----
*		csvPath						    : Path to the csv data.
*		normDataPath					: Path to the data.
*		outPath							: Path where the trained model will be saved.
*		normalized						: Type of data
*
***************************************************************************/
ReturnType FitFromDataRF(char *csvPath, char *dataPath, char *outPath, bool normalized) {

	ReturnType ret = { 0, "No Error" };
	
	try
	{
		/****************************************************/
		/*		     Import Normalized Data into matrix		*/
		/****************************************************/
		if(prop->verbose)
		{
			std::cout << *prop << std::endl;
		}
		//Read csv with image name + labels
		LabelsAndFeaturesData dat = readCSV(csvPath);
		//cv::Mat trainClasses = dat.matrix.t();
		std::vector<std::string> imgFileNames = dat.imgFileNames;
		cv::Mat features = dat.features;
		cv::Mat trainSamples;
		cv::Mat normTS;
		
		if (CreateDirectory(outPath, NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			if(prop->verbose)
				std::cout << "Taking features from path " << dataPath << std::endl;
			
			normTS = importFileFeatures(dataPath, prop->verbose, Constants::TOTAL_FEATURES);
			
			if(!normalized)
			{
				normTS = createNormalizationFile(outPath, normTS);
			}
			
			/****************************************************/
			/*					  Train RF						*/
			/****************************************************/
			
			// Construct the classifier and set the parameters
			//CvRTrees  rtrees[Constants::NUM_CLASSIFIERS][1];
			CvRTrees  rtrees[6][1];
			// Construct the classifier and set the parameters
			//CvRTrees** rtrees;
			//allocateRtrees(&rtrees,Constants::NUM_CLASSIFIERS,1);
			float priors[] = {1,1,1,1,1,1};
			CvRTParams  params( prop->max_depth,						// max_depth,
								prop->min_samples_count,				// min_sample_count,
								0.f,							// regression_accuracy,
								false,							// use_surrogates,
								prop->max_categories,					// max_categories,
								priors,								// priors,
								false,							// calc_var_importance,
								prop->nactive_vars,				// nactive_vars,
								prop->max_num_of_trees_in_forest, // max_num_of_trees_in_the_forest,
								0,								// forest_accuracy,
								CV_TERMCRIT_ITER				// termcrit_type
							   );

			// define all the attributes as numerical
			// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
			// that can be assigned on a per attribute basis
			cv::Mat var_type = cv::Mat(Constants::TOTAL_FEATURES + 1, 1, CV_8U );
			var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
			// this is a classification problem (i.e. predict a discrete number of class
			// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
			var_type.at<uchar>(Constants::TOTAL_FEATURES, 0) = CV_VAR_CATEGORICAL;
			/*cv::FileStorage file_norm("normTS.txt", cv::FileStorage::WRITE);
			file_norm << "normTS" << normTS;*/
			for(int i=0; i < Constants::NUM_CLASSIFIERS; i++){

				//Prepare trainClasses for the oneVsAll classification strategy
				//cv::Mat trainClass_i = oneVsAll(trainClasses, i);
				cv::Mat trainClass_i = dat.matrix.col(i);
				// Fit the classifier with the training data
				/*cv::FileStorage file_train("trainClass" + std::to_string((long double)i) + ".txt", cv::FileStorage::WRITE);
				file_train << "trainClass_i" << trainClass_i;*/
				std::cout << std::endl << "Training(" << i << ")...";
				clock_t start = clock();
				rtrees[i]->train( normTS, CV_ROW_SAMPLE, trainClass_i, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params );
				std::cout << "time:" <<  clock() - start << "ms" << std::endl;
				// Save Model
				char fileName[10000];
				sprintf(fileName, "%smodel_%d.xml", outPath, i);
				rtrees[i]->save(fileName);
			}
			//releaseRTrees(rtrees, Constants::NUM_CLASSIFIERS,1);
		} else {
			throw (std::string)"Folder " + outPath + " could not be created";
		}
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	catch(...)
	{
		std::cout << "General error" << std::endl;
	}
	return ret;
}

/**************************************************************************
*								PredictRF
*								---------
*		imagePath  : Path to the image we want to predict on.
*		modelPath  : Path to the trained model file.
*		n_bins	   : Number of histogram bins.
*		rad_grad   : Gradient Radius.
*		rad_dens   : Density Radius.
*		rad_entr   : Entropy Radius.
*
***************************************************************************/
ReturnType PredictRF(float** probs, unsigned char* data, int w, int h, const char* modelDir, void* handle, const float* features) {
	ReturnType ret = { 0, "No Error" };
	try
	{
		cv::Mat matFea = cv::Mat(1,Constants::NUM_FEATURES,CV_32F,(void*)features);
		/****************************************************/
		/*		     Load Image & Feature Extraction		*/
		/****************************************************/
		cv::Mat sample;
		cv::Mat normSample;
		sample = LoadImage_and_FeatExtr (data, w, h, matFea);
		normSample = readTrainedMeanStd(modelDir, sample);
		sample.release();

		/****************************************************/
		/*				    Predict Labels					*/
		/****************************************************/

		// Calculate prediction
		//float prediction[Constants::NUM_CLASSIFIERS];
		float *prediction = new float[Constants::NUM_CLASSIFIERS];

		// Initialice rtree;
		CvRTrees* rtree = (CvRTrees*)handle;
		for(int i=0; i<Constants::NUM_CLASSIFIERS; i++){

			prediction[i] = rtree[i].predict_prob(normSample);
		}
	
		*probs = prediction;
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	return ret;
}

ReturnType InitModel(CvRTrees** handle,const char *modelPath)
{
	ReturnType ret = { 0, "No Error" };
	clock_t begin;
	double load_rtree = 0;
	CvRTrees* models = new CvRTrees[Constants::NUM_CLASSIFIERS];
	for(int i=0; i<Constants::NUM_CLASSIFIERS; i++){
			// Initialice rtree;
			
		
			// Load the trained model
			char modelName[10000];
			sprintf(modelName,"%smodel_%i.xml", modelPath, i);
			if(_access(modelName, 0) != -1)
			{
				begin = clock();
				models[i].load(modelName);
				load_rtree += double(clock() - begin);
			}
			else
			{
				std::string err = "ERROR: file " + (std::string)modelName + " could not be opened. Is the path okay?";
				std::cerr << err << std::endl;
				throw std::exception(err.c_str());
			}
	}
	
	*handle = models;

	if(prop->verbose)
		std::cout << "Load Trees[" << load_rtree << "]" << std::endl;
	return ret;
}

ReturnType ReleaseModel(CvRTrees* handle)
{
	ReturnType ret = { 0, "No Error" };
	delete[] handle;
	return ret;
}

ReturnType CrossPredictRF(float** probs, void* handle, double* normalizedFeatures)
{
	ReturnType ret = { 0, "No Error" };
	try
	{
		cv::Mat normSample;
		clock_t global = clock();
		clock_t begin;
		
		normSample = cv::Mat(1,Constants::TOTAL_FEATURES,CV_32F,normalizedFeatures);
		

		/****************************************************/
		/*				    Predict Labels					*/
		/****************************************************/
		// Calculate prediction
		//float prediction[Constants::NUM_CLASSIFIERS];
		float* prediction = new float[Constants::NUM_CLASSIFIERS];
		double load_rtree = 0; 
		double predict_prob_time = 0;
		CvRTrees* rtrees = (CvRTrees*)handle;
		for(int i=0; i<Constants::NUM_CLASSIFIERS; i++){
			begin = clock();
			prediction[i] = rtrees[i].predict_prob(normSample);
			predict_prob_time += double(clock() - begin);
		}
	
		*probs = prediction;
		if(prop->verbose)
			std::cout << "Predict[" << predict_prob_time << "]" << std::endl;
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	
	return ret;
}

/**************************************************************************
*								ExtractFeatures
*								---------
*		csvPath				: Path to the file with the fingerPrint features.
*		imagesPath			: Path to the fingerprint image collection.
*		outPath				: Output path with the results {normalization, unnormalizedData, normalizedData}
***************************************************************************/
ReturnType ExtractFeatures(char* csvPath, char* imagesPath, char* outPath, char* prefix)
{
	ReturnType ret = { 0, "No Error" };
	try
	{
		//Read csv with image name + labels
		LabelsAndFeaturesData dat = readCSV(csvPath);
		//cv::Mat trainClasses = dat.matrix.t();
		std::vector<std::string> imgFileNames = dat.imgFileNames;
		cv::Mat features = dat.features;
		cv::Mat trainSamples;
		cv::Mat normTS;
		
		for(int i = 0; i != imgFileNames.size(); i++){
			clock_t time_a = clock();
			cv::Mat featurei = features.row(i);
			cv::Mat in = cv::imread(imagesPath + imgFileNames[i],cv::IMREAD_GRAYSCALE);
			cfg.fileName = (char*)imgFileNames[i].c_str();
			if(in.rows == 0) {
				throwError("ERROR: file " + (std::string)imagesPath + imgFileNames[i] + " could not be opened. Is the path okay?");
			}
			if(prop->verbose)
				std::cout << imgFileNames[i] << " img "  << i; 
			cv::Mat hist = LoadImage_and_FeatExtr(in.data, in.rows, in.cols, featurei,0);
			// Join features
			if(trainSamples.rows == 0)
				trainSamples = hist;
			else
				cv::vconcat(trainSamples, hist, trainSamples);
			hist.release();
			clock_t time_b = clock();
			if(prop->verbose)
				std::cout << "train. length: [" << trainSamples.rows << "," << trainSamples.cols  << "] time: " << (long)(time_b - time_a) << std::endl;
		}
		if(prop->verbose)
		{
			std::cout << *prop << std::endl;
		}
		exportFileFeatures(trainSamples, imgFileNames, ((std::string)outPath + "/" + prefix + "-raw.csv").c_str());
			
		normTS = createNormalizationFile(outPath,trainSamples);
		trainSamples.release();
		exportFileFeatures(normTS, imgFileNames, ((std::string)outPath + "/" + prefix + "-norm.csv").c_str());
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	
	return ret;
}

ReturnType ExportMeanStdFile(const char* unNormalizedDataPath, const char* outPath, bool verbose)
{
	ReturnType ret = { 0, "No Error" };
	try
	{
		cv::Mat trainSamples = importFileFeatures(unNormalizedDataPath, verbose, Constants::TOTAL_FEATURES);	
		createNormalizationFile(outPath,trainSamples);
	}
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	
	return ret;
}

std::ostream& operator<<(std::ostream& os, const Properties& prop)
{
    printParamsRF(prop);
    return os;
}

ReturnType SetProperties(Properties* param)
{
	ReturnType ret = { 0, "No Error" };	
	prop = param;
	if(prop->verbose)
		std::cout << "SetProperties ->" << std::endl << *prop << std::endl;
	return ret;
}

ReturnType ReleaseFloatPointer(float* pointer) 
{
	ReturnType ret = { 0, "No Error" };	
	if(pointer != NULL)
		delete [] pointer;
	return ret;
}



ReturnType PredictFromLabelsAndFeatureFile(const char* labelsPath, const char* imagesPath, const char* modelPath)
{
	ReturnType ret = { 0, "No Error" };	
	try
	{
		LabelsAndFeaturesData data = readCSV(labelsPath);
		std::vector<std::string> imgFileNames = data.imgFileNames;
		cv::Mat features = data.features;
		

		if(ret.code > 0)
			throw new std::exception(ret.message);
		
		CvRTrees* handle;
		InitModel(&handle,modelPath);
		for(int i = 0; i < imgFileNames.size(); i++)
		for(unsigned int i = 0; i < imgFileNames.size(); i++)
		{
			clock_t time_a = clock();
			cv::Mat featurei = features.row(i);
			std::cout << featurei << std::endl;
			cv::Mat in = cv::imread(imagesPath + imgFileNames[i],cv::IMREAD_GRAYSCALE);
			float *resultats;
			PredictRF(&resultats, in.data, in.cols, in.rows, modelPath, handle, (const float*)featurei.data);
			if(in.rows == 0) {
				std::string tmp = "ERROR: file " + (std::string)imagesPath + imgFileNames[i] + " could not be opened. Is the path okay?";
				throw new std::exception(tmp.c_str());
			}
			std::cout << imgFileNames[i] << " img "  << i;
		}
		ReleaseModel(handle);

	} 
	catch(std::exception& ex)
	{
		ret.code = 1;
		ret.message = ex.what();
	}
	return ret;
}