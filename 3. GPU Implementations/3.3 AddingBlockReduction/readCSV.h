/**********************************************************************
 *
 * readCSV.h -- readCSV function for reading the input from
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

/* Struct for Sparse Matrix type in the Compressed Sparse Row Format (CSR) */
typedef struct Sparse_Matrix_in_CSR_format {
   int 		nnz;
   // float* 	csrVal;   
   int* 	csrRowPtr;
   int* 	csrColInd;
}csrFormat;
/* The vector csrVal = [ 1 1 .... 1 ]; of length nnz is redundant 
 * as of the fact that it's all Aces ('1').
 * Only thing that matters is the positions of these Aces thus csrRowPtr 
 * alongside with csrColInd are adequate enough on their own */

#ifndef READ_CSV_H
#define READ_CSV_H

int readCSV(char* fName, csrFormat* A, int* N, int* M, int* nT_Mat, double* matlab_time);

#endif /* READ_CSV_H */