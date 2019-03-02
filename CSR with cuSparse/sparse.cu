#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <sys/time.h>
#include "readCSV.h"


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void mulSparse(csrFormat* A, csrFormat* C, int N){

    // Initialize cuSPARSE
    cusparseHandle_t handle;   
    cusparseCreate(&handle);

    int nnzA = A->nnz;

    // Allocate gpu memory to hold matrix in CSR format
    float* devVal;
    int* devCol, *devRowPtr;
    cudaMalloc((void**)&devVal,nnzA*sizeof(float));
    cudaMalloc((void**)&devCol,nnzA*sizeof(int));
    cudaMalloc((void**)&devRowPtr,(N+1)*sizeof(int));
    cudaMemcpy(devVal,A->csrVal,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A->csrColInd,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRowPtr,A->csrRowPtr,(N+1)*sizeof(int),cudaMemcpyHostToDevice);


    // Descriptor for sparse matrix A and C
    cusparseMatDescr_t descrA,descrC;     
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);

    // Row pointer vector for the product matrix in CSR format
    int* d_C_RowPtr;
    cudaMallocManaged(&d_C_RowPtr,(N+1)*sizeof(*d_C_RowPtr));

    // Calculate the number of C's non-zero values
    int nnzC;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,devRowPtr,devCol,descrA,nnzA,devRowPtr,devCol,descrC,d_C_RowPtr,&nnzC);

    printf("Non zero of C: %d\n",nnzC);


    // Allocate gpu memory to hold matrix C in CSR format
    float* d_C;
    int *d_C_ColIndices;
    cudaMalloc((void**)&d_C, (nnzC)*sizeof(*d_C));
    cudaMalloc((void**)&d_C_ColIndices, (nnzC) * sizeof(*d_C_ColIndices));
    
    // Calculate sparse array C
    double start = cpuSecond();
    cusparseScsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,devVal,devRowPtr,devCol,descrA,nnzA,devVal,devRowPtr,devCol,descrC,d_C,d_C_RowPtr,d_C_ColIndices);
    printf("Time elapsed for multiplication: %f seconds\n",cpuSecond()-start);

    // Return him by struct reference
    C->nnz = nnzC;
    C->csrVal = d_C;
    C->csrRowPtr = d_C_RowPtr;
    C->csrColInd =  d_C_ColIndices;


    cudaFree(devCol);
    cudaFree(devVal);
    cudaFree(devRowPtr);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrC);
    cusparseDestroy(handle);
}