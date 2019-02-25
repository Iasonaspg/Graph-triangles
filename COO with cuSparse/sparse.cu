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

    float* devVal;
    int* devCol, *devRow;
    cudaMallocManaged(&devVal,nnzA*sizeof(float));
    cudaMallocManaged(&devCol,nnzA*sizeof(int));
    cudaMallocManaged(&devRow,nnzA*sizeof(int));
    cudaMemcpy(devVal,A->cooValA,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A->cooColIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRow,A->cooRowIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);

    int* csrRowPtrA;
    cudaMallocManaged(&csrRowPtrA,(N+1)*sizeof(int));
    // cudaMemcpy(csrRowPtrA,A->cooRowIndA,(N+1)*sizeof(int),cudaMemcpyHostToDevice);
    cusparseXcoo2csr(handle,devRow,nnzA,N,csrRowPtrA,CUSPARSE_INDEX_BASE_ZERO);
    

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
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,csrRowPtrA,devCol,descrA,nnzA,csrRowPtrA,devCol,descrC,d_C_RowPtr,&nnzC);


    printf("Non zero of C: %d\n",nnzC);

    float* d_C;
    int *d_C_ColIndices;
    cudaMallocManaged(&d_C, (nnzC)*sizeof(*d_C));
    cudaMallocManaged(&d_C_ColIndices, (nnzC) * sizeof(*d_C_ColIndices));
    double start = cpuSecond();
    cusparseScsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,devVal,csrRowPtrA,devCol,descrA,nnzA,devVal,csrRowPtrA,devCol,descrC,d_C,d_C_RowPtr,d_C_ColIndices);
    printf("Time elapsed for multiplication: %f seconds\n",cpuSecond()-start);

    int* cooRowC;
    cudaMallocManaged(&cooRowC,(nnzC)*sizeof(int));

    cusparseXcsr2coo(handle,d_C_RowPtr,(nnzC),N,cooRowC,CUSPARSE_INDEX_BASE_ZERO);

    C->nnz = nnzC;
    C->cooValA = d_C;
    C->cooRowIndA = cooRowC;
    C->cooColIndA =  d_C_ColIndices;
}