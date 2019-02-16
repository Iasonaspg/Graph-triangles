#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cusparse_v2.h>

int main()
{
    // Initialize cuSPARSE
    cusparseHandle_t handle;   
    cusparseCreate(&handle);

    /**************************/
    /* SETTING UP THE PROBLEM */
    /**************************/
    const int N     = 3;                // Number of rows and columns

    // Host side dense matrices
    double *h_A_dense = (double*)malloc(N * N * sizeof(double));
    double *h_x_dense = (double*)malloc(N *     sizeof(double));
    double *h_y_dense = (double*)malloc(N *     sizeof(double));

    // Column-major ordering
    h_A_dense[0] = 0.4612;  h_A_dense[4] = -2.06;     h_A_dense[8]  = 10.35; 
    h_A_dense[1] = -3.06; h_A_dense[5] = 0;      
    h_A_dense[2] = 1.3566;  h_A_dense[6] = 7.07;       
    h_A_dense[3] = 0;      h_A_dense[7] = 0.0;          

    // --- Initializing the data and result vectors
    for (int k = 0; k < N; k++) {
        h_x_dense[k] = 1.;
        h_y_dense[k] = 0.;
    }

    // --- Create device arrays and copy host arrays to them
    double *d_A_dense;  
    cudaMalloc(&d_A_dense, N*N*sizeof(double));
    double *d_x_dense;  
    cudaMalloc(&d_x_dense, N*sizeof(double));
    double *d_y_dense;  
    cudaMalloc(&d_y_dense, N*sizeof(double));
    cudaMemcpy(d_A_dense, h_A_dense, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_dense, h_x_dense, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_dense, h_y_dense, N*sizeof(double), cudaMemcpyHostToDevice);

    // Descriptor for sparse matrix A
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
    cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA);

    // Host side number of nonzero elements per row of matrix A
    int *h_nnzPerVectorA = (int *)malloc(N*sizeof(*h_nnzPerVectorA));
    cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost);

    //printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
    for (int i = 0; i < N; ++i){
        //printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
    } 
    printf("\n");

    // Device side sparse matrix
    double *d_A, *d_C;            
    cudaMalloc(&d_A, nnzA*sizeof(*d_A));
    int *d_A_RowIndices, *d_C_RowIndices;    
    cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices));
    cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices));
    int *d_A_ColIndices, *d_C_ColIndices;    
    cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices));

    cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices);

    int nnzC = 0;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseXcsrgemmNnz(handle,transA,transA,N,N,N,descrA,nnzA,d_A_RowIndices,d_A_ColIndices,descrA,nnzA,d_A_RowIndices,d_A_ColIndices,descrC,d_C_RowIndices,&nnzC);

    printf("Non zero of C: %d\n",nnzC);

    cudaMalloc(&d_C, nnzC*sizeof(*d_C));
    cudaMalloc(&d_C_ColIndices, nnzC * sizeof(*d_C_ColIndices));
    cusparseDcsrgemm(handle,transA,transA,N,N,N,descrA,nnzA,d_A,d_A_RowIndices,d_A_ColIndices,descrA,nnzA,d_A,d_A_RowIndices,d_A_ColIndices,descrC,d_C,d_C_RowIndices,d_C_ColIndices);

    // Host side sparse matrices
    double *h_C = (double *)malloc(nnzC * sizeof(*h_C));        
    // int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
    // int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
    cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost);

    // printf("\nOriginal matrix A in CSR format\n\n");
    for (int i = 0; i < nnzC; ++i) printf("C[%i] = %f\n", i, h_C[i]); printf("\n");

    // printf("\n");
    // for (int i = 0; i < (N + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

    // printf("\n");
    // for (int i = 0; i < nnzA; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);  

    // printf("\n");
    // for (int i = 0; i < N; ++i) printf("h_x[%i] = %f \n", i, h_x_dense[i]); printf("\n");

    const double alpha = 1.;
    const double beta  = 0.;
    cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzA, &alpha, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_x_dense,&beta, d_y_dense);

    cudaMemcpy(h_y_dense,d_y_dense,N*sizeof(double),cudaMemcpyDeviceToHost);

    // printf("\nResult vector\n\n");
    // for (int i = 0; i < N; ++i){
    //     printf("h_y[%i] = %f\n", i, h_y_dense[i]);
    // }


    return 0;
}