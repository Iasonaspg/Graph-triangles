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


// int main()
// {
//     // Initialize cuSPARSE
//     cusparseHandle_t handle;   
//     cusparseCreate(&handle);

//     const int N = 3;                // Number of rows and columns

//     // Host side dense matrices
//     float *h_A_dense = (float*)malloc(N * N * sizeof(float));
//     float *h_x_dense = (float*)malloc(N *     sizeof(float));
//     float *h_y_dense = (float*)malloc(N *     sizeof(float));

//     // Column-major ordering
//     h_A_dense[0] = 1;  h_A_dense[4] = 6;     h_A_dense[8]  = 5; 
//     h_A_dense[1] = 0; h_A_dense[5] = 0;      
//     h_A_dense[2] = 1;  h_A_dense[6] = 4;       
//     h_A_dense[3] = 0;      h_A_dense[7] = 0.0;          

//     // Initializing the data and result vectors
//     for (int k = 0; k < N; k++) {
//         h_x_dense[k] = 1.;
//         h_y_dense[k] = 0.;
//     }

//     // Create device arrays and copy host arrays to them
//     float *d_A_dense;  
//     cudaMalloc(&d_A_dense, N*N*sizeof(float));
//     float *d_x_dense;  
//     cudaMalloc(&d_x_dense, N*sizeof(float));
//     float *d_y_dense;  
//     cudaMalloc(&d_y_dense, N*sizeof(float));
//     cudaMemcpy(d_A_dense, h_A_dense, N*N*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_x_dense, h_x_dense, N*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y_dense, h_y_dense, N*sizeof(float), cudaMemcpyHostToDevice);


//     // Descriptor for sparse matrix A and C
//     cusparseMatDescr_t descrA,descrC;     
//     cusparseCreateMatDescr(&descrA);
//     cusparseCreateMatDescr(&descrC);
//     //cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
//     //cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);  

//     int nnzA = 0;                           // Number of nonzero elements in dense matrix A

//     const int lda = N;                      // Leading dimension of dense matrix

//     // Device side number of nonzero elements per row of matrix A
//     int *d_nnzPerVectorA;   
//     cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA));
//     cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA);

//     // Sparse matrix
//     float *d_A;            
//     cudaMallocManaged(&d_A, nnzA*sizeof(*d_A));
//     int *d_A_RowIndices;    
//     cudaMallocManaged(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices));
//     int *d_A_ColIndices;    
//     cudaMallocManaged(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices));

//     cusparseSdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices);

//     cooFormat* test = (cooFormat*)malloc(sizeof(cooFormat));
//     test->nnz = nnzA;
//     test->cooValA = d_A;
//     test->cooColIndA = d_A_ColIndices;
//     test->cooRowIndA = d_A_RowIndices;
    
//     // for (int i=0;i<N+1;i++){
//     //     printf("Row: %d\n",d_A_RowIndices[i]);
//     // }
//     // for (int i=0;i<nnzA;i++){
//     //     printf("Col: %d\n",d_A_ColIndices[i]);
//     // }

//     //printf("Random value: %f\n",d_A[3]);
//     cooFormat* test1 = (cooFormat*)malloc(sizeof(cooFormat));
//     // cudaMallocManaged(&test1,sizeof(csrFormat));

//     mulSparse(test,test1,N);
    

//     for (int i=0; i<test1->nnz;i++){
//         // printf("Values: %f\n",test1->csrVal[i]);
//     }
    
//     // const float alpha = 1.;
//     // const float beta  = 0.;
//     // cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzA, &alpha, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_x_dense,&beta, d_y_dense);

//     // cudaMemcpy(h_y_dense,d_y_dense,N*sizeof(double),cudaMemcpyDeviceToHost);

//     // printf("\nResult vector\n\n");
//     // for (int i = 0; i < N; ++i){
//     //     printf("h_y[%i] = %f\n", i, h_y_dense[i]);
//     // }


//     return 0;
// }

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
    

    for (int i=0;i<N+1;i++){
        // printf("csrValA from func: %f\n",devVal[i]);
        // printf("csrRowPtrA from func: %d\n",csrRowPtrA[i]);
        //printf("csrColIndA from func: %d\n",devCol[i]);
    }
    for (int i=0;i<nnzA;i++){
        // printf("csrColIndA from func: %d for value: %f\n",devCol[i],devVal[i]);
    }

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

    printf("timi: %d\n",(nnzC));
    C->nnz = nnzC;
    C->cooValA = d_C;
    C->cooRowIndA = cooRowC;
    C->cooColIndA =  d_C_ColIndices;
}