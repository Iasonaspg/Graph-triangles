/**********************************************************************
 *
 * readCSV.c -- readCSV function for reading the input from
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>  
#include <inttypes.h>
#include <errno.h>
#include <math.h>
#include "readCSV.h"

int split_line_int(char* str, char* delim, int* args);
int split_line_float(char* str, char* delim, float* args);
char *trim_space(char *in);
int readCSV(char* fName, csrFormat* A, cooFormat* A_COO, int* N, int* M, int* nT_Mat, double* matlab_time);
// int findLines(char* fName);

/*
int main(int argc, char** argv){

    char* fName = argv[1];
    int N, M, nT_Mat;
    double matlab_time;

    csrFormat A;
    cooFormat A_COO;

    readCSV(fName, &A, &A_COO, &N, &M, &nT_Mat, &matlab_time);

    printf("Input Data File Sample:\n");    
    printf("nnz = %d\n", A.nnz);
    for (int i=0;i<10;i++){
        printf("csrVal: %f\n",A.csrVal[i]);
        printf("csrRowPtr: %d\n",A.csrRowPtr[i]);
        printf("csrColInd: %d\n",A.csrColInd[i]);
    }

    printf("Validation File:\n N = %d, M = %d\n Matlab result was %d, produced in %lf\n", N, M, nT_Mat, matlab_time);
    
    return 0; 
}
*/


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

int readCSV(char* fName, csrFormat* A, cooFormat* A_COO, int* N, int* M, int* nT_Mat, double* matlab_time){

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    /* Constructing the full .csv file names */
    char *csvFileName, *csvFileNameCOO, *valFileName;
    csvFileName = (char*)malloc(1000*sizeof(char));
    csvFileNameCOO = (char*)malloc(1000*sizeof(char));
    valFileName = (char*)malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Path to file: ~/PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(csvFileName,  "/home/johnfli/Code/PD_4/Data/DataDIMACS10_");
    // Do not change "DataDIMACS10_" unless you want to give it as input name aintside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(csvFileName, fName);    

    strcat(valFileName, csvFileName);
    strcat(valFileName, "_validation_file.csv");

    strcpy(csvFileNameCOO, csvFileName);
    strcat(csvFileNameCOO, "_COO.csv"); 
 
    strcat(csvFileName, ".csv"); 

    /* Reading the original matrix N and M, as well as 
       the matlab result and time elapsed */
    fp = fopen(valFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    if( (read = getline(&line, &len, fp)) != -1 ) {
        char* token = strtok(line, ",");
        char* endPtr;

        (*N) = strtoimax(token,&endPtr,10);
        
        token = strtok(NULL, ",");
        
        (*M) = strtoimax(token,&endPtr,10);

        A->nnz = 2 * (*M);   A_COO->nnz = A->nnz;

        token = strtok(NULL, ",");

        (*nT_Mat) = strtoimax(token,&endPtr,10);

        token = strtok(NULL, ",");   

        (*matlab_time) = atof(token);
    }

    /* Close file */
    fclose(fp);

    /* Allocating memory to hold the struct of Sparse Matrix A */
    A->csrVal = (float*)malloc ((A->nnz)*sizeof(float));
    A->csrRowPtr = (int*)malloc (((*N)+1)*sizeof(int));
    A->csrColInd = (int*)malloc ((A->nnz)*sizeof(int));

    /* Reading the input data */
    fp = fopen(csvFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

     if ((read = getline(&line, &len, fp)) != -1)
        split_line_float(line,",",A->csrVal);

     if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrRowPtr);

     if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrColInd);
        
    /* Close file */
    fclose(fp);

    /* Allocating memory to hold the struct of Sparse Matrix A_COO */
    A_COO->cooVal = (float*)malloc ((A_COO->nnz)*sizeof(float));
    A_COO->cooRowInd = (int*)malloc ((A_COO->nnz)*sizeof(int));
    A_COO->cooColInd = (int*)malloc ((A_COO->nnz)*sizeof(int));

    /* Reading the input data in COO Format */
    fp = fopen(csvFileNameCOO, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    char* token;
    char* endPtr;

    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1){
        
        token = strtok(line, ",");

         A_COO->cooVal[i] = atof(token);

        token = strtok(NULL, ",");

        A_COO->cooRowInd[i] = strtoimax(token,&endPtr,10);        

        token = strtok(NULL, ",");

        A_COO->cooColInd[i] = strtoimax(token,&endPtr,10);

        i++;

     }
        
    /* Close file */
    fclose(fp);

    return EXIT_SUCCESS;
}

int split_line_int(char* str, char* delim, int* tmp){
    int i = 0;
    char* token = strtok(str, delim);
    char* strNum;
    char* endPtr;
    while (token != NULL) {
        strNum = trim_space(token);

        tmp[i++] = strtoimax(strNum,&endPtr,10);

        token = strtok(NULL, delim);   
    }
    return i;
}

int split_line_float(char* str, char* delim, float* tmp){
    int i = 0;
    char* token = strtok(str, delim);
    while (token != NULL) {

        tmp[i++] = atof(token);

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
