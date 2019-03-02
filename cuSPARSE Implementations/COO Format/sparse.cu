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

void mulSparse(cooFormat* A, cooFormat* C, int N){

    // Initialize cuSPARSE
    cusparseHandle_t handle;   
    cusparseCreate(&handle);

    int nnzA = A->nnz;

    // Bring array to gpu memory
    float* devVal;
    int* devCol, *devRow;
    cudaMalloc((void**)&devVal,nnzA*sizeof(float));
    cudaMalloc((void**)&devCol,nnzA*sizeof(int));
    cudaMalloc((void**)&devRow,nnzA*sizeof(int));
    cudaMemcpy(devVal,A->cooValA,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A->cooColIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRow,A->cooRowIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);

    // Convert to CSR format for cuSparse needs
    int* csrRowPtrA;
    cudaMallocManaged(&csrRowPtrA,(N+1)*sizeof(int));
    cusparseXcoo2csr(handle,devRow,nnzA,N,csrRowPtrA,CUSPARSE_INDEX_BASE_ZERO);
    

    // Descriptor for sparse matrix A and C
    cusparseMatDescr_t descrA,descrC;     
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);

    int* d_C_RowPtr;
    cudaMalloc((void**)&d_C_RowPtr,(N+1)*sizeof(*d_C_RowPtr));

    // Calculate the number of C's non-zero values
    int nnzC;
    //cudaMallocManaged(&nnzC,sizeof(int));
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,csrRowPtrA,devCol,descrA,nnzA,csrRowPtrA,devCol,descrC,d_C_RowPtr,&nnzC);

    printf("Non zero of C: %d\n",nnzC);

    // Allocate memory for array C
    float* d_C;
    int *d_C_ColIndices;
    cudaMalloc((void**)&d_C, (nnzC)*sizeof(*d_C));
    cudaMalloc((void**)&d_C_ColIndices, (nnzC) * sizeof(*d_C_ColIndices));
    
    // Calculate the multiplication A*A
    double start = cpuSecond();
    cusparseScsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,devVal,csrRowPtrA,devCol,descrA,nnzA,devVal,csrRowPtrA,devCol,descrC,d_C,d_C_RowPtr,d_C_ColIndices);
    printf("Time elapsed for multiplication: %f seconds\n",cpuSecond()-start);

    // Convert row array from CSR to COO format
    int* cooRowC;
    cudaMalloc((void**)&cooRowC,(nnzC)*sizeof(int));
    cusparseXcsr2coo(handle,d_C_RowPtr,(nnzC),N,cooRowC,CUSPARSE_INDEX_BASE_ZERO);

    C->nnz = nnzC;
    C->cooValA = d_C;
    C->cooRowIndA = cooRowC;
    C->cooColIndA =  d_C_ColIndices;

    cudaFree(devCol);
    cudaFree(devVal);
    cudaFree(devRow);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrC);
    cusparseDestroy(handle);
}