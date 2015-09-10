#include <Windows.h>
#include <iostream>
#include <fstream>
#include "FpFeaturesLibrary.h"
#include "FoldsSplitter.h"
#include "utils.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\operations.hpp"
#include <time.h>

LList *getDirFiles(char *dir);
void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos);
void show_histogram(std::string const& name, cv::Mat const& image);
bool writeHistToFile(std::string name, cv::Mat* data);

std::string GetElapsedTime(clock_t time_a, clock_t time_b) 
{
	if (time_a == ((clock_t)-1) || time_b == ((clock_t)-1))
	{
		return "Unable to calculate elapsed time";
	}
	else
	{
		long total_time_ticks = (long)(time_b - time_a);
		return std::to_string((long double)total_time_ticks);
	}	
}
using namespace cv;
int main(int argc, char** argv) {
	
	/*uchar data[25] = {255,255,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	Mat* grayMat = new cv::Mat(5,5,CV_8U,data);
	
	std::cout << entropy(grayMat,new Mat()) << std::endl;*/

	LList *files = getDirFiles((char *)"\\\\ssd2015\\Data\\FpFeatures_Comparison\\input\\");
	
	
	if (files != NULL){
		LList *fp = files;
		Config* cfg = new Config();
		cv::FileStorage file_grad;
		cv::FileStorage file_dens;
		long total_time_ticks;
		int i = 0;
		while (fp != NULL){
			cv::Mat in;
			cv::Mat out_grad;
			try
			{
				FName data = fp->element;
				in = cv::imread((char *)data.fpath, cv::IMREAD_GRAYSCALE);
			
				std::string pout = "\\\\ssd2015\\Data\\FpFeatures_Comparison\\opencv\\";
				
				
				cfg->path = (char*)pout.c_str();
				cfg->fileName = (char*)data.fname;
				cfg->verboseHough = true;
				setConfig(cfg);
				

				std::cout << "imatge (" << i << ") " << cfg->fileName << std::endl;
				
				/*********************************/
				diferentiate_img(&in);
				/*********************************/
				
				clock_t time_a = clock();
				out_grad = hist_grad(&in, 1, 32);
				clock_t time_b = clock();
				
				total_time_ticks = (long)(time_b - time_a);
				
				/*if(i % 1000 == 0)
				{*/
					file_grad = cv::FileStorage((const std::string)pout + fp->element.fname + "_grad.txt", cv::FileStorage::WRITE);
					file_grad << "GradHist" << out_grad;
				/*}*/
				
				std::cout << "\tGradHist OK..." + GetElapsedTime(time_a,time_b) + "ms" << std::endl;
				
				time_a = clock();
				cv::Mat out_dens = hist_density(&in,3,32);
				time_b = clock();
				total_time_ticks += (long)(time_b - time_a);
				
				/*if(i % 1000 == 0)
				{*/
					file_dens = cv::FileStorage((const std::string)pout + fp->element.fname + "_dens.txt", cv::FileStorage::WRITE);
					file_dens << "DensHist" << out_dens;
				/*}*/
				std::cout << "\tDensHist OK..." + GetElapsedTime(time_a,time_b) + "ms" << std::endl;
				
				time_a = clock();
				cv::Mat out_hough = hist_hough(&in,32);
				time_b = clock();
				total_time_ticks += (long)(time_b - time_a);
				
				/*if(i % 1000 == 0)
				{*/
					cv::FileStorage file_hough((const std::string)pout + fp->element.fname + "_hough.txt", cv::FileStorage::WRITE);
					file_hough << "HoughHist" << out_hough;
				/*}*/
				
				std::cout << "\tHoughHist OK..." + GetElapsedTime(time_a,time_b) + "ms" << std::endl;

				time_a = clock();

				cv::Mat out_entropy = hist_entropy(&in,5,32);
				time_b = clock();
				total_time_ticks += (long)(time_b - time_a);
				
				/*if(i % 1000 == 0)
				{*/
					cv::FileStorage file_entropy((const std::string)pout + fp->element.fname + "_entropy.txt", cv::FileStorage::WRITE);
					file_entropy << "EntropyHist" << out_entropy;
				/*}*/
				std::cout << "\tEntropyHist OK..." + GetElapsedTime(time_a,time_b) + "ms" << std::endl;


				std::cout << std::to_string((long double)total_time_ticks) << std::endl;
				
				i++;
				fp = fp->next;
				out_grad.release();
				file_grad.release();
				out_dens.release();
				file_dens.release();
			}
			
			catch (const cv::Exception& ex)
			{
				std::cout << ex.what() << std::endl;
			}
			catch (...)
			{
				std::exception_ptr p = std::current_exception();
				std::cout << "Error genérico" << std::endl;
			}
		}
		system("pause");
		delete cfg;
		files->free();
	}
	return 0;
}

void show_histogram(std::string const& name, cv::Mat const& image)
{
	// Set histogram bins count
	int bins = 32;
	int histSize[] = { bins };
	// Set ranges for histogram bins
	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };

	// create matrix for histogram visualization
	int const hist_height = 256;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);

	// visualize each bin
	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int   const height = cvRound(binVal*hist_height / max_val);
		cv::line
			(hist_image
			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			, cv::Scalar::all(255)
			);
	}
	cv::imshow(name, hist_image);
}
/*
int main(int argc, const char* argv[])
{
	// here you can use cv::IMREAD_GRAYSCALE to load grayscale image, see image2
	cv::Mat3b const image1 = cv::imread("C:\\workspace\\horse.png", cv::IMREAD_COLOR);
	cv::Mat1b image1_gray;
	cv::cvtColor(image1, image1_gray, cv::COLOR_BGR2GRAY);
	cv::imshow("image1", image1_gray);
	show_histogram("image1 hist", image1_gray);

	cv::Mat1b const image2 = cv::imread("C:\\workspace\\bunny.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("image2", image2);
	show_histogram("image2 hist", image2);

	cv::waitKey();
	return 0;
}*/
//** *****************************************************************
//** ** HELPER METHODS
//** *****************************************************************

/* getDirFiles(dir)
*
* Lists all of the files inside of the path specified by 'dir' and returns them
* inside of a linked list structure, where each element of the linked list is a
* structure containing the name and path of the file.
*
*     dir:    Pointer to a character array specifying the path where the files
*             are to be found.
*/
LList *getDirFiles(char *dir){
	LList *ret = NULL, *cpos = NULL;
	char *sst;

	// List PNG files
	sst = strconcat(dir, "*.png");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List JPG files
	sst = strconcat(dir, "*.jpg");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List BIN files
	sst = strconcat(dir, "*.bin");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	return ret;
}

void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos){
	WIN32_FIND_DATA ffd;

	// Get handle to first file
	HANDLE hFind = FindFirstFile((char *)pattern, &ffd);
	if (hFind == INVALID_HANDLE_VALUE) return;

	// Build linked list with file paths
	bool hasNext = true;
	while (hasNext){
		if (*ret == NULL){
			*ret = new LList;
			*cpos = *ret;
		}
		else{
			(*cpos)->next = new LList;
			*cpos = (*cpos)->next;
		}

		(*cpos)->element.fname = strclone(ffd.cFileName);
		(*cpos)->element.fpath = strconcat(dir, ffd.cFileName);
		hasNext = FindNextFile(hFind, &ffd);
	}
}