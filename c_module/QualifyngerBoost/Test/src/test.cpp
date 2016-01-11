#include <windows.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "FoldSplitter.h"
#include "utils.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\operations.hpp"
#include "opencv2\ml\ml.hpp"
#include "ApiDS.h"
#include <time.h>
#include <numeric>

using namespace std;
using namespace cv;

LList *getDirFiles(char *dir);
void getMatchedFiles(char *dir, char *pattern, LList **ret, LList **cpos);
int digitalizeFingers(Mat* output, const Mat onlyFingerNumber);
int createNorm(const Mat, Mat*);
int genKFold(Mat*, Mat*, int, int, bool);
int filter(Mat*, Mat, Mat);
int eval_pred(Mat*, const Mat, const Mat);
template<typename T> int saveMatToCSV(Mat, const char*);
void norm_y(Mat*);


//int main(int argc, char* argv[])
//{
//	Utils::verbose = true;
//	SetVerbose(Utils::verbose); 
//	CsvData X_out_train;
//	CsvData	X_out_test;
//	CsvData	y_out_train;
//	CsvData	y_out_test;
//	
//	Mat ind_train;
//	Mat ind_test;
//
//	int r = genKFold(&ind_train, &ind_test, 1179855, 10, true);
//	saveMatToCSV<int>(ind_train.row(0).t(), "ind_train.csv");
//
//	
//	clock_t begin, end;
//	begin = clock();
//	Utils::loadCSV(&y_out_train, "y_out_train.csv", ';', 0, 1, false);
//	end = clock();
//	cout << "Load y_out_train..." << double(end - begin) / CLOCKS_PER_SEC << "sec" << endl;
//	begin = clock();
//	Utils::loadCSV(&X_out_train, "X_out_train.csv", ';', 0, 22, false);
//	end = clock();
//	cout << "Load X_out_train..." << double(end - begin) / CLOCKS_PER_SEC << "sec" << endl;
//	Utils::loadCSV(&y_out_test, "y_out_test.csv", ';', 0, 1, false);
//	Utils::loadCSV(&X_out_test, "X_out_test.csv", ';', 0, 22, false);
//	
//
//	vector<Stump> weak_class_arr;
//	Mat agg_class_est_train;
//
//	ReturnType ret = Train(&weak_class_arr, &agg_class_est_train, X_out_train.body, y_out_train.body, 40, 10);
//	if (ret.code > 0)
//		throw new exception(ret.message);
//}

//int main(int argc, char* argv[])
//{
//	vector<Stump> v;
//	
//	Stump v1;
//	v1.alpha = 2.2f;
//	v1.dim = 8;
//	v1.ineq = LESS_THAN;
//	v1.thresh = 1.28f;
//	Stump v2;
//	v2.alpha = 4.4f;
//	v2.dim = 16;
//	v2.ineq = BIGGER_THAN;
//	v2.thresh = 2.56f;
//	v.push_back(v1);
//	v.push_back(v2);
//
//
//	WriteToFile(&v, "Stump.txt");
//	Stump out;
//	ReadFromFile(&v, "Stump.txt");
//}

//int main(int argc, char* argv[])
//{
//	Mat m1 = Mat(3, 3, CV_32F, Scalar(1));
//	Mat m2 = Mat(3, 3, CV_32F, Scalar(1));
//	for (int i = 0; i < m1.rows; i++)
//		for (int j = 0; j < m1.cols; j++)
//		{ 
//			m1.at<float>(i, j) *= sqrt(i*m1.rows + j);
//			m2.at<float>(i, j) = (i*m1.rows + j) % 2;
//		}
//	cout << m1 << endl;
//	cout << m2 << endl;
//	cout << m1.dot(m2) << endl;
//	
//
//
//
//	//float a[] = { 0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
//	bool b[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0 };
//
//	float* a = (float*)m1.data;
//	//float* b = (float*)m2.data;
//
//	cout << inner_product(a, a+m1.rows*m1.cols, b, 0.0) << endl;
//}

int main(int argc, char* argv[])
{
	float thresh = 1.3f;
	char* pathCsv = "//ssd2015/DataFase1/Empremptes/CSVs/Totes.csv";
	if (argc > 1)
	{
		thresh = (float)atof(argv[1]);
		if (argc > 2)
			pathCsv = argv[2];
	}
	Utils::verbose = true;
	SetVerbose(Utils::verbose);
	int ret = 0;
	clock_t begin,end;
	
	CsvData data;
	
	cout << "Load CSV " << pathCsv << " ..." << endl;
	begin = clock();
	ret = Utils::loadCSV(&data, pathCsv, ';', 3, 19, true);
	end = clock();
	cout << double(end - begin) / CLOCKS_PER_SEC << "sec" << endl;
	
	Mat geyce_y, kit4_y;
	vector<string>::iterator it;
	it = find(data.headers.begin(), data.headers.end(), "GEYCEResult");
	int ind = (int)distance(data.headers.begin(), it);
	geyce_y = data.body.col(ind);
	
	it = find(data.headers.begin(), data.headers.end(), "SAGEMResult");
	ind = (int)distance(data.headers.begin(), it);
	kit4_y = data.body.col(ind);
	
	data.body = data.body.colRange(0, 13);

	it = find(data.headers.begin(), data.headers.end(), "dedo");
	int posDedo = (int)distance(data.headers.begin(), it);
	Mat digFingers;
	digitalizeFingers(&digFingers, data.body.col(posDedo));
	
	Mat output = data.body.colRange(0, posDedo);
	if (output.cols == 0)
		output = data.body.colRange(posDedo + 1, data.body.cols);
	else
		hconcat(output, data.body.colRange(posDedo + 1, data.body.cols), output);
	hconcat(output, digFingers, output);

	Mat norMat;
	saveMatToCSV<float>(output, "QFB_X.csv");
	createNorm(output, &norMat);
	saveMatToCSV<float>(norMat, "QF_X_Norm.csv");
	
	Mat ind_train;
	Mat ind_test;

	ret = genKFold(&ind_train, &ind_test, norMat.rows, 10, true);
	
	vector<int> TP, TP_geyce;
	vector<int> TN, TN_geyce;
	vector<int> FP, FP_geyce;
	vector<int> FN, FN_geyce;

	for (int i = 0; i < 10; i++)
	{	
		if (Utils::verbose)
			cout << "Fold " << i + 1 << "/ 10" << endl << "-----------" << endl;
		Mat X_out_train;
		Mat	X_out_test;
		Mat	y_out_train;
		Mat	y_out_test;

		Mat geyce_y_out_test;

		//Debug purposes
		filter(&X_out_train, data.body, ind_train.row(i));
		filter(&X_out_test, data.body, ind_test.row(i));
		saveMatToCSV<float>(X_out_train, "QF_X_train.csv");
		saveMatToCSV<float>(X_out_test, "QF_X_test.csv");
		//End Debug


		filter(&X_out_train, norMat, ind_train.row(i));
		filter(&X_out_test, norMat, ind_test.row(i));
		filter(&y_out_train, kit4_y, ind_train.row(i));
		filter(&y_out_test, kit4_y, ind_test.row(i));
		filter(&geyce_y_out_test, geyce_y, ind_test.row(i));

		vector<Stump> weak_class_arr;
		Mat agg_class_est_train;
		ReturnType ret;
 		ret = Train(&weak_class_arr, &agg_class_est_train, X_out_train, y_out_train, 40, 10);
		if (ret.code > 0)
			throw new exception(ret.message);
		
		Mat y_out_test_pred;
		Mat agg_class_est_test;
		ret = Test(&y_out_test_pred, &agg_class_est_test, X_out_test, &weak_class_arr, thresh);
		if (ret.code > 0)
			throw new exception(ret.message);
		
		Mat contingency_table;
		eval_pred(&contingency_table, y_out_test_pred, y_out_test);

		TP.push_back(contingency_table.at<int>(0, 0));
		TN.push_back(contingency_table.at<int>(0, 1));
		FP.push_back(contingency_table.at<int>(0, 2));
		FN.push_back(contingency_table.at<int>(0, 3));

		cout << "\nGeyce\'s Evaluation\n------------------\n" << endl;
		
		Mat contingency_table_geyce;
		eval_pred(&contingency_table_geyce, geyce_y_out_test, y_out_test);
		TP_geyce.push_back(contingency_table_geyce.at<int>(0, 0));
		TN_geyce.push_back(contingency_table_geyce.at<int>(0, 1));
		FP_geyce.push_back(contingency_table_geyce.at<int>(0, 2));
		FN_geyce.push_back(contingency_table_geyce.at<int>(0, 3));
	}
}

//** *****************************************************************
//** ** HELPER METHODS
//** *****************************************************************



void norm_y(Mat* y)
{
	for (int i = 0; i < y->rows; i++)
		for (int j = 0; j < y->cols; j++)
		{
			float v = y->at<float>(i, j);
			if (v == 0)
				y->at<float>(i, j) = -1;
			else
				y->at<float>(i, j) = 1;
		}
}

/*
"""
Evaluates a prediction with the real labels and return
a list of proportion of true positives, true negatives,
false positives and false negatives.

: type pred_y : npumpy.ndarray of int8
: param pred_y : Predicted binary labels
: type y : npumpy.ndarray of int8
: param y : Real binary labels

: rtype : npumpy.ndarray
: return : [true_pos, true_neg, false_pos, false_neg]
"""
*/
int eval_pred(Mat* contingency_table, const Mat pred_y, const Mat y)
{
	Mat tmp = Mat(1, 4, CV_32S);
	int true_pos = 0;
	int	true_neg = 0;
	int false_pos = 0;
	int	false_neg = 0;

	for (int i = 0; i < pred_y.rows; i++)
	{
		int y_i = y.at<float>(i, 0);
		int pred_y_i = pred_y.at<float>(i, 0);
		if (y_i == pred_y_i)
		{
			if (y_i > 0)
				true_pos++;
			else
				true_neg++;
		}
		else
		{
			if (pred_y_i > 0)
				false_pos++;
			else
				false_neg++;
		}
	}

	float acc = (true_pos + true_neg) / (float)y.rows;
	float err = (false_pos + false_neg) / (float)y.rows;
	float NPV = true_neg / float(true_neg + false_neg + 1e-16);
	float FPR = false_pos / float(false_pos + true_neg + 1e-16);
	float pre = true_pos / float(true_pos + true_neg + 1e-16);
	float rec = true_pos / float(true_pos + false_neg + 1e-16);
	float f1;
	if (pre + rec != 0)
		f1 = 2 * pre*rec / (pre + rec);
	else
		f1 = 0;

	cout << "\tAccuracy: " << acc << "\n\tError Rate: " << err << "\n\tPrecision:" 
		<< pre << "\n\tRecall: " << rec << "\n\tF1 Score: " << f1 << "\n\tNPV:" << NPV << "\n\tFPR: " << FPR <<
		"\n\tTP:" << true_pos << "\n\tTN: " << true_neg << "\n\tFP: " << false_pos << "\n\tFN: " << false_neg << endl;
	
	tmp.at<int>(0, 0) = true_pos;
	tmp.at<int>(0, 1) = true_neg;
	tmp.at<int>(0, 2) = false_pos;
	tmp.at<int>(0, 3) = false_neg;

	*contingency_table = tmp;
	
	return 0;
}
/*
assert pred_y.shape[0] == len(y), 'The legth of both label sets must be equal.'
n = len(y)

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for i in range(n) :
if y[i] == pred_y[i] :
if y[i] > 0:
true_pos += 1
else :
true_neg += 1
else:
if pred_y[i] > 0:
false_pos += 1
else :
false_neg += 1

if verbose :
acc = (true_pos + true_neg) / float(n)
err = (false_pos + false_neg) / float(n)
NPV = true_neg / float(true_neg + false_neg + 1e-16)
FPR = false_pos / float(false_pos + true_neg + 1e-16)
pre = true_pos / float(true_pos + true_neg + 1e-16)
rec = true_pos / float(true_pos + false_neg + 1e-16)
f1 = 2 * pre*rec / (pre + rec) if pre + rec != 0 else 0
print '\tAccuracy: {0}\n\tError Rate: {1}\n\tPrecision: ' \
'{2}\n\tRecall: {3}\n\tF1 Score: {4}\n\tNPV: {5}\n\tFPR: {6}\n' \
'\tTP: {7}\n\tTN: {8}\n\tFP: {9}\n\tFN: {10}\n'.format(acc, err, pre, rec, f1,
NPV, FPR, true_pos, true_neg,
false_pos, false_neg)
return np.array([true_pos, true_neg, false_pos, false_neg])
*/

int filter(Mat* output, const Mat input, const Mat indexs)
{
	Mat tmp;
	for (int i = 0; i < indexs.cols; i++)
	{
		int index = indexs.at<int>(0, i);
		Mat nextRow = input.row(index);
		if (tmp.rows > 0)
			tmp.push_back(nextRow);
		else
			tmp = nextRow;
	}
	*output = tmp;
	return 0;
}

/*
*	Generates K-Fold indexs, train_it with random numbers
*
*	it: number of partitions
*	size: size of the input
*/
int genKFold(Mat* train_it, Mat* test_it, int size, int nPart, bool shuffled)
{
	int test_size = size / nPart;
	int train_size = size - size / nPart;
	test_size = test_size + (size - test_size - train_size);

	Mat tmp_train = Mat(nPart, train_size, CV_32S, -1);
	Mat tmp_test = Mat(nPart, test_size, CV_32S, -1);
	
	for (int i = 0; i < nPart; i++)
	{
		int test_begin = i*test_size;
		int test_end = (i + 1)*test_size;
		int test_ind = 0;
		int train_ind = 0;
		int* shuffled_train_index = new int[train_size];
		for (int n = 0; n < train_size; n++) shuffled_train_index[n] = n;
		random_shuffle(&shuffled_train_index[0], &shuffled_train_index[train_size - 1]);
		for (int j = 0; j < size; j++)
		{
			if (j >= test_begin && j < test_end)
				tmp_test.at<int>(i, test_ind++) = j;
			else
			{
				if (shuffled)
					tmp_train.at<int>(i, shuffled_train_index[train_ind++]) = j;
				else
					tmp_train.at<int>(i, train_ind++) = j;
			}
				
		}
		delete[] shuffled_train_index;
	}

	*train_it = tmp_train;
	*test_it = tmp_test;
	return 0;
}

template<typename T> int saveMatToCSV(Mat mat, const char* outFile)
{
	string fileName = outFile;
	ofstream myfile(fileName);
	if (myfile.is_open())
	{
		for (int i = 0; i < mat.rows; i++)
		{
			//myfile << imgPaths[i];
			myfile << mat.at<T>(i, 0);
			for (int j = 1; j < mat.cols; j++)
				myfile << "," << mat.at<T>(i, j);
			myfile << "\n";
		}
		myfile.close();
	}
	else
	{
		throw new exception(("Unable to open file " + fileName).c_str());
	}
}

int createNorm(const Mat input, Mat* output)
{
	Mat nzed;

	cv::Mat temp1, temp2, mean, std, norm_i;
	nzed = cv::Mat(input.size(), input.type());
	
	for (int i = 0; i < input.cols; i++)
	{
		cv::meanStdDev(input.col(i), mean, std);
		//mean.convertTo(mean,CV_32F);
		//std.convertTo(std,CV_32F);
		cv::subtract(input.col(i), mean, temp1);
		cv::divide(temp1, std, temp2);
		norm_i = nzed.colRange(i, i + 1);
		temp2.copyTo(norm_i);
	}
	*output = nzed;
	return 0;
}

int digitalizeFingers(Mat* output, const Mat onlyFingerNumber)
{
	Mat tmp = Mat(onlyFingerNumber.rows, 10, onlyFingerNumber.type());
	try
	{
		for (int i = 0; i < tmp.rows; i++)
		{
			int nFinger = (int)onlyFingerNumber.at<float>(i, 0);
			for (int j = 0; j < tmp.cols; j++)
			{
				if (j + 1 != nFinger)
					tmp.at<float>(i, j) = 0;
				else
					tmp.at<float>(i, j) = 1;
			}
		}
	}
	catch (...)
	{
		return -1;
	}
	*output = tmp;
	return 0;
}


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
	sst = Utils::strconcat(dir, "*.png");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List JPG files
	sst = Utils::strconcat(dir, "*.jpg");
	getMatchedFiles(dir, sst, &ret, &cpos);
	delete[] sst;

	// List BIN files
	sst = Utils::strconcat(dir, "*.bin");
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
	BOOL hasNext = true;
	while (hasNext){
		if (*ret == NULL){
			*ret = new LList;
			*cpos = *ret;
		}
		else{
			(*cpos)->next = new LList;
			*cpos = (*cpos)->next;
		}

		(*cpos)->element.fname = Utils::strclone(ffd.cFileName);
		(*cpos)->element.fpath = Utils::strconcat(dir, ffd.cFileName);
		hasNext = FindNextFile(hFind, &ffd);
	}
}