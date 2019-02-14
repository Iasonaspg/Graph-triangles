#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include <errno.h>
#include <math.h>


int split(char* str, char* delim, long* args);
char *trim_space(char *in);
void readCSV(char* fName, long* arr, long* tmp);
int findLines(char* fName);

int main(int argc, char** argv){

    char* fName = argv[1];

    int leng = 3*findLines(fName); 
    printf("Lines in the file are: %d\n",leng);   
    
    long* arr = malloc(leng*sizeof(long));
    long* tmp = malloc(3*sizeof(long));
    
    readCSV(fName,arr,tmp);
    
    for (int i=0;i<9;i++){
        // printf("Arr: %ld\n",arr[i]);
    }

    return 0; 
}


int findLines(char* fName){

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(fName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }
    
    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        i++;
    }
    fclose(fp);
    return i;
}

void readCSV(char* fName, long* arr, long* tmp){

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(fName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        // printf("%s", line);
        split(line,",",tmp);
        // printf("I: %d\n",i);
        for (int j=0;j<3;j++){
            arr[i*3+j] = tmp[j];
            // printf("Temp: %ld\n",tmp[j]);
        }
        i++;
    }
    fclose(fp);
}

int split(char* str, char* delim, long* tmp){
    int i = 0;
    char* token = strtok(str, delim);
    char* strNum;
    char* endPtr;
    while (token != NULL) {
        strNum = trim_space(token);
        // printf("Number: %ld\n", strtoimax(strNum,&endPtr,10));
        tmp[i++] = strtoimax(strNum,&endPtr,10);
        // printf("Number: %ld\n", tmp[i-1]);
        token = strtok(NULL, delim);   
    }
    return i;
}

char *trim_space(char *in){
    char *out = NULL;
    int len;
    if (in) {
        len = strlen(in);
        while(len && isspace(in[len - 1])) --len;
        while(len && *in && isspace(*in)) ++in, --len;
        if (len) {
            out = strndup(in, len);
        }
    }
    return out;
}
