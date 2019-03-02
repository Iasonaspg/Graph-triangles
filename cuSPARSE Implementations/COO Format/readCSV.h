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

/* Struct for Sparse Matrix type in the Coordinate Format (COO) */
typedef struct Sparse_Matrix_in_COO_format {
   int   nnz;
   float* cooValA;
   int* cooRowIndA;
   int* cooColIndA;
}cooFormat;

int readCSV(char* fName, struct Sparse_Matrix_in_COO_format *A, int* N, int* M, int* nT_Mat, double* matlab_time);
int split(char* str, char* delim, long* args);
char *trim_space(char *in);
int findLines(char* fName);
void mulSparse(cooFormat* A, cooFormat* C, int N);
int findTrianglesCPU(cooFormat* A, cooFormat* C);
double cpuSecond();
__global__ void filter(cooFormat A, cooFormat C, int* counter1, int* counter2);
__global__ void findTriangles(cooFormat A, cooFormat C, int* sum);
__global__ void findTrianglesShared(cooFormat A, cooFormat C, int* totalSum, int* counter);
__global__ void findTrianglesSum(cooFormat A, cooFormat C, int* totalSum, int* counter);

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
