#ifndef FOLDS_SPLITTER_H
#define FOLDS_SPLITTER_H

// Structure that stores the name and path of files
struct FName{
    char *fname;
    char *fpath;
};

// Single-Linked list structure
struct LList{
    FName element;
    LList *next;
    
    LList() : next(0){};
    
    void free(){
        LList *list = this, *tlist;
        while(list != 0){
            tlist = list;
            list = tlist->next;
            delete tlist;
        }
    }
};

class FoldsSplitter{
	int a_numFolds;
	LList **a_training;
	LList **a_testing;

	public:
		FoldsSplitter(LList *list, int nFolds);
		LList *getFoldTraining(int idx);
		LList *getFoldTesting(int idx);
};

#endif