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
int readCSV(char* fName, struct Sparse_Matrix_in_CSR_format *A, int* N, int* M, int* nT_Mat, double* matlab_time);
// int findLines(char* fName);


int main(int argc, char** argv){

    char* fName = argv[1];
    int N, M, nT_Mat;
    double matlab_time;

    struct Sparse_Matrix_in_CSR_format A;

    readCSV(fName, &A, &N, &M, &nT_Mat, &matlab_time);

    printf("Input Data File Sample:\n");    
    printf("nnz = %d\n", A.nnz);
    for (int i=0;i<10;i++){
        printf("csrValA: %f\n",A.csrValA[i]);
        printf("csrRowPtrA: %d\n",A.csrRowPtrA[i]);
        printf("csrColIndA: %d\n",A.csrColIndA[i]);
    }

    printf("Validation File:\n N = %d, M = %d\n Matlab result was %d, produced in %lf\n", N, M, nT_Mat, matlab_time);
    
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

int readCSV(char* fName, struct Sparse_Matrix_in_CSR_format *A, int* N, int* M, int* nT_Mat, double* matlab_time){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    //int* tmp = malloc(3*sizeof(int));

    /* Constructing the full .csv file names */
    char* csvFileName;
    csvFileName = malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Path to file: ~/PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(csvFileName,  "/home/johnfli/Code/PD_4/Data/DataDIMACS10_");
    // Do not change "DataDIMACS10_" unless you want to give it as input name aintside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(csvFileName, fName);    

    char* valFileName;
    valFileName = malloc(1000*sizeof(char));
    strcat(valFileName, csvFileName);
    strcat(valFileName, "_validation_file.csv");

    strcat(csvFileName, ".csv");

    // int leng = findLines(csvFileName);

    // printf("Lines in the file are: %d\n",--leng);   

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

    /* Allocating memory to hold struct of sparse matrix A */
    A->csrValA = malloc ((A->nnz)*sizeof(float));
    A->csrRowPtrA = malloc (((*N)+1)*sizeof(int));
    A->csrColIndA = malloc ((A->nnz)*sizeof(int));

    /* Reading the input data */
    fp = fopen(csvFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

     if ((read = getline(&line, &len, fp)) != -1)
        split_line_float(line,",",A->csrValA);
        // printf("%s", line);
     if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrRowPtrA);
        // printf("I: %d\n",i);
     if ((read = getline(&line, &len, fp)) != -1)
        split_line_int(line,",",A->csrColIndA);
        // printf("A[%d]: %d\n", j, A[j]);
        
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
        // printf("Number: %d\n", strtoimax(strNum,&endPtr,10));
        tmp[i++] = strtoimax(strNum,&endPtr,10);
        // printf("Number: %d\n", tmp[i-1]);
        token = strtok(NULL, delim);   
    }
    return i;
}

int split_line_float(char* str, char* delim, float* tmp){
    int i = 0;
    char* token = strtok(str, delim);
    // char* strNum;
    // char* endPtr;
    while (token != NULL) {
        // strNum = trim_space(token);
        // printf("Number: %d\n", strtoimax(strNum,&endPtr,10));
        tmp[i++] = atof(token);
        // printf("Number: %d\n", tmp[i-1]);
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
