#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <sys/time.h>


typedef struct Sparse_Matrix_in_CSR_format {
   int   nnz;
   float* csrVal;
   int* csrRowPtr;
   int* csrColInd;
}csrFormat;

void mulSparse(csrFormat* A, csrFormat* C, int N);

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void pri(float* tmp){
    printf("Random value: %f\n",tmp[4]);
}

int main()
{
    // Initialize cuSPARSE
    cusparseHandle_t handle;   
    cusparseCreate(&handle);

    const int N = 3;                // Number of rows and columns

    // Host side dense matrices
    float *h_A_dense = (float*)malloc(N * N * sizeof(float));
    float *h_x_dense = (float*)malloc(N *     sizeof(float));
    float *h_y_dense = (float*)malloc(N *     sizeof(float));

    // Column-major ordering
    h_A_dense[0] = 0.4612;  h_A_dense[4] = -2.06;     h_A_dense[8]  = 10.35; 
    h_A_dense[1] = -3.06; h_A_dense[5] = 0;      
    h_A_dense[2] = 1.3566;  h_A_dense[6] = 7.07;       
    h_A_dense[3] = 0;      h_A_dense[7] = 0.0;          

    // Initializing the data and result vectors
    for (int k = 0; k < N; k++) {
        h_x_dense[k] = 1.;
        h_y_dense[k] = 0.;
    }

    // Create device arrays and copy host arrays to them
    float *d_A_dense;  
    cudaMalloc(&d_A_dense, N*N*sizeof(float));
    float *d_x_dense;  
    cudaMalloc(&d_x_dense, N*sizeof(float));
    float *d_y_dense;  
    cudaMalloc(&d_y_dense, N*sizeof(float));
    cudaMemcpy(d_A_dense, h_A_dense, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_dense, h_x_dense, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_dense, h_y_dense, N*sizeof(float), cudaMemcpyHostToDevice);


    // Descriptor for sparse matrix A and C
    cusparseMatDescr_t descrA,descrC;     
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);
    //cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    //cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);  

    int nnzA = 0;                           // Number of nonzero elements in dense matrix A

    const int lda = N;                      // Leading dimension of dense matrix

    // Device side number of nonzero elements per row of matrix A
    int *d_nnzPerVectorA;   
    cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA));
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA);

    // Sparse matrix
    float *d_A;            
    cudaMallocManaged(&d_A, nnzA*sizeof(*d_A));
    int *d_A_RowIndices;    
    cudaMallocManaged(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices));
    int *d_A_ColIndices;    
    cudaMallocManaged(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices));

    cusparseSdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices);

    csrFormat* test = (csrFormat*)malloc(sizeof(csrFormat));
    test->nnz = nnzA;
    test->csrVal = d_A;
    test->csrColInd = d_A_ColIndices;
    test->csrRowPtr = d_A_RowIndices;
    
    //printf("Random value: %f\n",d_A[3]);
    csrFormat* test1 = (csrFormat*)malloc(sizeof(csrFormat));
    // cudaMallocManaged(&test1,sizeof(csrFormat));

    mulSparse(test,test1,N);
    

    for (int i=0; i<test1->nnz;i++){
        printf("Values: %f\n",test1->csrVal[i]);
    }
    
    // const float alpha = 1.;
    // const float beta  = 0.;
    // cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzA, &alpha, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_x_dense,&beta, d_y_dense);

    // cudaMemcpy(h_y_dense,d_y_dense,N*sizeof(double),cudaMemcpyDeviceToHost);

    // printf("\nResult vector\n\n");
    // for (int i = 0; i < N; ++i){
    //     printf("h_y[%i] = %f\n", i, h_y_dense[i]);
    // }


    return 0;
}

void mulSparse(csrFormat* A, csrFormat* C, int N){

    // Initialize cuSPARSE
    cusparseHandle_t handle;   
    cusparseCreate(&handle);

    // csrFormat* devA;  
    // cudaMalloc((void**)&devA,sizeof(csrFormat));  
    // cudaMemcpy(devA,A,sizeof(csrFormat),cudaMemcpyHostToDevice);

    // printf("Inside func: %f\n",A->csrVal[4]);

    // Descriptor for sparse matrix A and C
    cusparseMatDescr_t descrA,descrC;     
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrC);

    int* d_C_RowPtr;
    cudaMallocManaged((void**)&d_C_RowPtr,(N+1)*sizeof(*d_C_RowPtr));

    int nnzA = A->nnz;

    int nnzC = 0;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,A->csrRowPtr,A->csrColInd,descrA,nnzA,A->csrRowPtr,A->csrColInd,descrC,d_C_RowPtr,&nnzC);


    printf("Non zero of C: %d\n",nnzC);

    float* d_C;
    int *d_C_ColIndices;
    cudaMallocManaged(&d_C, nnzC*sizeof(*d_C));
    cudaMallocManaged(&d_C_ColIndices, nnzC * sizeof(*d_C_ColIndices));
    double start = cpuSecond();
    cusparseScsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,A->csrVal,A->csrRowPtr,A->csrColInd,descrA,nnzA,A->csrVal,A->csrRowPtr,A->csrColInd,descrC,d_C,d_C_RowPtr,d_C_ColIndices);
    printf("Time elapsed for generating points: %f seconds\n",cpuSecond()-start);

    C->nnz = nnzC;
    C->csrVal = d_C;
    C->csrRowPtr = d_C_RowPtr;
    C->csrColInd =  d_C_ColIndices;
}