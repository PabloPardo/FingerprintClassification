#ifndef UTILS_H
#define UTILS_H

void memcpy(char *orig, char *dest, long size);
long strlen(char *str);
char *strclone(char *str);
char *strconcat(char *str1, char *str2);
void int2str(int val, int minLen, char *ret);
char *dbl2str(double val, int minLen);
void dbl2str(double val, int minLen, char *ret);
double str2dbl(char *string);

#endif /* UTILS_H */