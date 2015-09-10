#ifndef FA_TYPES_H
#define FA_TYPES_H
#include <fstream>
#include <iostream>
struct Config {
	bool verboseGrad;
	bool verboseHough;
	bool verboseDens;
	bool verboseEntropy;
	bool verboseDiff;
	char* path;
	char* fileName;
	Config() {
		verboseGrad = false;
		verboseHough = false;
		verboseDens = false;
		verboseEntropy = false;
		verboseDiff = false;
		fileName = NULL;
	}

	bool Config::writeMatToFile(char* title, cv::Mat* data)
	{
		std::string fileNameResult;
		fileNameResult = (std::string)path + title + fileName + ".txt";
		
		 cv::FileStorage file(fileNameResult, cv::FileStorage::WRITE);
		 file << (std::string)title << *data;
		return true;

	}
	
	bool Config::writeValueToFile(char* prefix, double data, char* title=NULL)
	{
		std::string fileNameResult;
		fileNameResult = (std::string)path + prefix + fileName + ".txt";
		std::ofstream fout(fileNameResult,std::ofstream::out | std::ofstream::app); //opening an output stream for file test.txt
		//checking whether file could be opened or not. If file does not exist or don't have write permissions, file
		//stream could not be opened.
		if (fout.is_open())
		{
			if(title != NULL)
				fout << title << std::endl;
		
			fout << data << std::endl;
			//file opened successfully so we are here
			//cout << "File Opened successfully!!!. Writing data from array to file" << endl;
		}
		else //file could not be opened
		{
			//cout << "File could not be opened." << endl;
			return false;
		}
		return true;
	}
};

#endif