/**********************************************************************
 *
 * readCSV.cu -- readCSV function for reading the input from
 *               a .csv file
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

int readCSV_COO(char* fName, cooFormat* A_COO, int* N, int* M, int* nT_Mat, double* matlab_time){

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    int* tmp = (int*) malloc(2*sizeof(int));

    /* Constructing the full .csv file names */
    char *valFileName;
    valFileName = (char*)malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Relative path to file: ../PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(valFileName,  "../../Data/DIMACS10_");
    // Do not change "DIMACS10_" unless you want to give it as input name aintside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(valFileName, fName);    
    strcat(valFileName, "_validation_file.csv");

    /* Reading the original matrix N and M, as well as 
       the matlab result and time elapsed */
    fp = fopen(valFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    if( ( getline(&line, &len, fp)) != -1 ) {
        char* token = strtok(line, ",");
        char* endPtr;

        (*N) = strtoimax(token,&endPtr,10);
        
        token = strtok(NULL, ",");
        
        (*M) = strtoimax(token,&endPtr,10);

        A_COO->nnz = 2 * (*M);

        token = strtok(NULL, ",");

        (*nT_Mat) = strtoimax(token,&endPtr,10);

        token = strtok(NULL, ",");   

        (*matlab_time) = atof(token);
    }

    /* Close file */
    fclose(fp);

    /* Constructing the full .csv file name */
    char *csvFileNameCOO_A;
    csvFileNameCOO_A = (char*)malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Relative path to file: ../PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(csvFileNameCOO_A,  "../../Data COO Format/DIMACS10_");
    // Do not change "DIMACS10_" unless you want to give it as input name aintside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(csvFileNameCOO_A, fName);
    strcat(csvFileNameCOO_A, "_COO.csv"); 

    /* Allocating memory to hold the struct of Sparse Matrix A */
    A_COO->cooVal =  (float*)malloc ((A_COO->nnz)*sizeof(float));
    A_COO->cooRowInd = (int*)malloc ((A_COO->nnz)*sizeof(int));
    A_COO->cooColInd = (int*)malloc ((A_COO->nnz)*sizeof(int));

    /* Reading the input data in COO Format */
    fp = fopen(csvFileNameCOO_A, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    int i;
    for (i = 0; i < A_COO->nnz; i++)
        A_COO->cooVal[i] = 1.0;

    i = 0;
    while (( getline(&line, &len, fp)) != -1){

        split_line_int(line,",",tmp);

        A_COO->cooRowInd[i] = tmp[0];       

        A_COO->cooColInd[i++] = tmp[1];
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
