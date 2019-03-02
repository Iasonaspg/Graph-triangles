/**********************************************************************
 *
 * readCSV.cu (.c) -- readCSV function for reading the input from
 *                    a .csv file
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


int readCSV(char* fName, csrFormat* A, int* N, int* M, int* nT_Mat, double* matlab_time){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    /* Constructing the full .csv file names */
    char *csvFileName, *valFileName;
    csvFileName = (char*)malloc(1000*sizeof(char));
    valFileName = (char*)malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Path to file: ~/PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(csvFileName,  "./Data/DIMACS10_");
    // Do not change "DataDIMACS10_" unless you want to give it as input name aintside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(csvFileName, fName);    

    strcat(valFileName, csvFileName);
    strcat(valFileName, "_validation_file.csv");

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

        A->nnz = 2 * (*M);

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

    for (int i=0;i<A->nnz;i++){
        A->csrVal[i] = 1;
    }

    /* Reading the input data */
    fp = fopen(csvFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrRowPtr);

    if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrColInd);
        
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
 