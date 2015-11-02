#include "FoldSplitter.h"

FoldsSplitter::FoldsSplitter(LList *list, int nFolds){
	this->a_numFolds = nFolds;
	this->a_training = (LList **)new LList *[nFolds];
	this->a_testing = (LList **)new LList *[nFolds];

	// Count number of elements in the list
	int nFiles = 0;
	LList *fp = list;
	while (fp != 0){
		nFiles++;
		fp = fp->next;
	}

	// Get number of elements per fold
	int fSize = nFiles / nFolds;

	for (int i = 0; i<nFolds; i++){
		LList *ttrain = new LList();
		LList *ttest = new LList();
		this->a_training[i] = ttrain;
		this->a_testing[i] = ttest;

		int nTrain = 0;
		int nTest = 0;
		fp = list;

		// Add first training part
		for (int j = 0; j<fSize*i; j++){
			ttrain->element = fp->element;
			nTrain++;
			if (nTrain < fSize*(nFolds - 1)){
				ttrain->next = new LList();
				ttrain = ttrain->next;
				fp = fp->next;
			}
		}

		// Add testing part
		for (int j = fSize*i; j<fSize*(i + 1); j++){
			ttest->element = fp->element;
			nTest++;
			if (nTest < fSize){
				ttest->next = new LList();
				ttest = ttest->next;
				fp = fp->next;
			}
		}

		// Add second training part
		for (int j = fSize*(i + 1); j<nFiles; j++){
			ttrain->element = fp->element;
			nTrain++;
			if (nTrain < fSize*(nFolds - 1)){
				ttrain->next = new LList();
				ttrain = ttrain->next;
				fp = fp->next;
			}
		}
	}
}

LList *FoldsSplitter::getFoldTraining(int idx){
	return this->a_training[idx];
}

LList *FoldsSplitter::getFoldTesting(int idx){
	return this->a_testing[idx];
}