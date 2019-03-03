/**********************************************************************
 *
 * readCSV.c -- readCSV function for reading the input from
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef READ_CSV_H
#define READ_CSV_H

/* Struct for Sparse Matrix type in the Coordinate Format (CSR) */
typedef struct Sparse_Matrix_in_CSR_format {
   int   nnz;
   float* csrVal;
   int* csrRowPtr;
   int* csrColInd;
}csrFormat;

int readCSV(char* fName, csrFormat *A, int* N, int* M, int* nT_Mat, double* matlab_time);
int split_line_int(char* str, char* delim, int* args);
char *trim_space(char *in);
int findLines(char* fName);


#endif /* READ_CSV_H */
