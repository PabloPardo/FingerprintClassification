#include "utils.h"

//** *****************************************************************
//** ** STRING MANIPULATION METHODS
//** *****************************************************************

void memcpy(char *orig, char *dest, long size){
    for(long i=0;i<size;i++){
        dest[i] = orig[i];
    }
}

long strlen(char *str){
    long ret = 0;
    while(str[ret] != 0) ret++;
    return ret;
}

char *strconcat(char *str1, char *str2){
    long len1 = strlen(str1);
    long len2 = strlen(str2);
    long len = len1 + len2 + 1;
    
    char *ret = new char[len];
    for(long i=0;i<len1;i++) ret[i] = str1[i];
    for(long i=0;i<len2;i++) ret[len1+i] = str2[i];
    ret[len-1] = 0;
    
    return ret;
}

char *strclone(char *str){
    long len = strlen(str) + 1;
    char *ret = new char[len];
    
    for(long i=0;i<len;i++) ret[i] = str[i];
    return ret;
}

void int2str(int val, int minLen, char *ret){
	double tval = ((double)val)+0.01;

	// Fix number of significative digits
	for(int i=1;i<minLen;i++){
		tval /= 10;
	}

	// Print digits
	int pos = 0;
	for(int i=0;i<minLen;i++){
		ret[pos] = '0' + ((int)tval);
		tval = (tval - ((int)tval)) * 10;
		pos++;
	}

	ret[pos] = '\0';
}

char *dbl2str(double val, int minLen){
	char *ret = new char[10];
	dbl2str(val, minLen, ret);
	return ret;
}

void dbl2str(double val, int minLen, char *ret){
	val += 0.0000001;
	int pos = 0;

	int nint = 0;
	while(val >= 1){
		nint++;
		val /= 10;
	}

	if(nint == 0){
		ret[pos] = '0';
		pos++;
	}else{
		while(pos < nint){
			val = (val - ((int)val)) * 10;
			ret[pos] = '0' + (int)val;
			pos++;
		}
	}

	if(val > 0){
		ret[pos] = '_';
		pos++;
		while(val > 0 && pos < minLen){
			val = (val - ((int)val)) * 10;
			ret[pos] = '0' + (int)val;
			pos++;
		}
	}

	while(pos < minLen){
		ret[pos] = '0';
		pos++;
	}

	ret[pos] = '\0';
}

double str2dbl(char *string){
	double value = 0;

	bool is_decimal = false;
	double dec_factor = 1;

	char *chr = string;
	while(chr[0] != '\0'){
		if(chr[0] == '.'){
			is_decimal = true;
		}else if(is_decimal){
			dec_factor *= 10;
			value = value + ((double)(chr[0] - '0'))/dec_factor;
		}else{
			value = value*10 + (chr[0] - '0');
		}

		chr = &chr[1];
	}

	return value;
}