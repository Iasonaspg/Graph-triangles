/**********************************************************************
 *
<<<<<<< HEAD
 * readCSV.c -- readCSV function for reading the input from
=======
 * readCSV.h -- readCSV function for reading the input from
>>>>>>> origin/john
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

<<<<<<< HEAD
/* Struct for Sparse Matrix type in the Coordinate Format (COO) */
typedef struct Sparse_Matrix_in_COO_format {
   int   nnz;
   float* cooValA;
   int* cooRowIndA;
   int* cooColIndA;
}cooFormat;
=======
/* Struct for Sparse Matrix type in the Compressed Sparse Row Format (CSR) */
typedef struct Sparse_Matrix_in_CSR_format {
   int 		nnz;
   float* 	csrVal;
   int* 	csrRowPtr;
   int* 	csrColInd;
}csrFormat;
>>>>>>> origin/john

#ifndef READ_CSV_H
#define READ_CSV_H

<<<<<<< HEAD
int readCSV(char* fName, struct Sparse_Matrix_in_COO_format *A, int* N, int* M, int* nT_Mat, double* matlab_time);
int split(char* str, char* delim, long* args);
char *trim_space(char *in);
int findLines(char* fName);
void mulSparse(cooFormat* A, cooFormat* C, int N);
double cpuSecond();

#define CHECK(call) \
{                    \
    const cudaError_t error = call; \
    if (error != cudaSuccess){       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);  \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \

#endif /* READ_CSV_H */
=======
int readCSV(char* fName, csrFormat* A, int* N, int* M, int* nT_Mat, double* matlab_time);

#endif /* READ_CSV_H */
>>>>>>> origin/john
