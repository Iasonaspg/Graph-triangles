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

    float* devVal;
    int* devCol, *devRowPtr;
    cudaMallocManaged(&devVal,nnzA*sizeof(float));
    cudaMallocManaged(&devCol,nnzA*sizeof(int));
    cudaMallocManaged(&devRowPtr,(N+1)*sizeof(int));
    cudaMemcpy(devVal,A->csrVal,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A->csrColInd,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRowPtr,A->csrRowPtr,(N+1)*sizeof(int),cudaMemcpyHostToDevice);

    

    // for (int i=0;i<N+1;i++){
    //     printf("csrValA from func: %f\n",devVal[i]);
    //     printf("csrRowPtrA from func: %d\n",csrRowPtrA[i]);
    //     printf("csrColIndA from func: %d\n",devCol[i]);
    // }
    // for (int i=0;i<nnzA;i++){
    //     printf("csrColIndA from func: %d for value: %f\n",devCol[i],devVal[i]);
    // }

    // printf("Inside func: %f\n",A->csrVal[4]);

    // Descriptor for sparse matrix A and C
    cusparseMatDescr_t descrA,descrC;     
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);

    int* d_C_RowPtr;
    cudaMallocManaged(&d_C_RowPtr,(N+1)*sizeof(*d_C_RowPtr));

    int nnzC;
    //cudaMallocManaged(&nnzC,sizeof(int));
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,devRowPtr,devCol,descrA,nnzA,devRowPtr,devCol,descrC,d_C_RowPtr,&nnzC);


    printf("Non zero of C: %d\n",nnzC);

    float* d_C;
    int *d_C_ColIndices;
    cudaMallocManaged(&d_C, (nnzC)*sizeof(*d_C));
    cudaMallocManaged(&d_C_ColIndices, (nnzC) * sizeof(*d_C_ColIndices));
    double start = cpuSecond();
    cusparseScsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,devVal,devRowPtr,devCol,descrA,nnzA,devVal,devRowPtr,devCol,descrC,d_C,d_C_RowPtr,d_C_ColIndices);
    printf("Time elapsed for multiplication: %f seconds\n",cpuSecond()-start);

    
    C->nnz = nnzC;
    C->csrVal = d_C;
    C->csrRowPtr = d_C_RowPtr;
    C->csrColInd =  d_C_ColIndices;
}