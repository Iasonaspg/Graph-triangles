/********************************************************************************
 *
 * sparse.cu -- Tester function for the csrgemm() CUDA function
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <sys/time.h>
#include "cuTrianglesFinder.h"
#include "readCSV.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char** argv){

    int N, M, nT_Mat;
    double matlab_time;

    /* Create the struct of type csr Format to hold the Sparse Matrices A and B */
    csrFormat h_A, d_A, 
              h_B, d_B;

    /* Create the struct of type coo Format to hold the Sparse Matrix A */
    cooFormat h_A_COO, d_A_COO;
    int* d_csrRowPtr_coo2csr;

    /* Read the input Sparse Matrix, alongside with some further info */
    readCSV(argv[1], &h_A, &h_A_COO, &N, &M, &nT_Mat, &matlab_time);

    printf("Input Data File Sample:\n");    
    printf("nnz = %d\n", h_A.nnz);
    for (int i=0;i<10;i++){
        printf("h_A.csrVal: %f\n",h_A.csrVal[i]);
        printf("h_A.csrRowPtr: %d\n",h_A.csrRowPtr[i]);
        printf("h_A.csrColInd: %d\n",h_A.csrColInd[i]);
    }

    printf("COO: Input Data File Sample:\n");    
    printf("nnz = %d\n", h_A_COO.nnz);
    for (int i=0;i<10;i++){
        printf("h_A_COO.cooVal: %f\n",h_A_COO.cooVal[i]);
        printf("h_A_COO.cooRowInd: %d\n",h_A_COO.cooRowInd[i]);
        printf("h_A_COO.cooColInd: %d\n",h_A_COO.cooColInd[i]);
    }

    /* Create the cuSPARSE handle */
    cusparseHandle_t handle = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));    

    /* Construct a descriptor of the matrix A */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    /* Construct a descriptor of the matrix A_COO */
    cusparseMatDescr_t descrA_COO = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA_COO));
    CHECK_CUSPARSE(cusparseSetMatType(descrA_COO, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA_COO, CUSPARSE_INDEX_BASE_ZERO));

    /* Construct a descriptor of the matrix B */
    cusparseMatDescr_t descrB = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate device memory to store the sparse CSR representation of A */
    d_A.nnz = h_A.nnz;
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrVal),    sizeof(float) * d_A.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrRowPtr), sizeof(int) * (N + 1)));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrColInd), sizeof(int) * d_A.nnz));

    /* Copy the sparse CSR representation of A from the Host to the Device */
    CUDA_CALL(cudaMemcpy(d_A.csrVal,    h_A.csrVal,    d_A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A.csrRowPtr, h_A.csrRowPtr, (N + 1) * sizeof(int)  , cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A.csrColInd, h_A.csrColInd, d_A.nnz * sizeof(int)  , cudaMemcpyHostToDevice));

    /* Allocate device memory to store the sparse COO representation of A */
    d_A_COO.nnz = h_A_COO.nnz;
    CUDA_CALL(cudaMalloc((void **)&(d_A_COO.cooVal),    sizeof(float) * d_A_COO.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_A_COO.cooRowInd), sizeof(int) * d_A_COO.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_csrRowPtr_coo2csr), sizeof(int) * (N + 1)));
    CUDA_CALL(cudaMalloc((void **)&(d_A_COO.cooColInd), sizeof(int) * d_A_COO.nnz));

    /* Copy the sparse COO representation of A from the Host to the Device */
    CUDA_CALL(cudaMemcpy(d_A_COO.cooVal,    h_A_COO.cooVal,    d_A_COO.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A_COO.cooRowInd, h_A_COO.cooRowInd, d_A_COO.nnz * sizeof(int)  , cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A_COO.cooColInd, h_A_COO.cooColInd, d_A_COO.nnz * sizeof(int)  , cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseXcoo2csr(handle,
                    d_A_COO.cooRowInd,
                    d_A_COO.nnz,
                    N,
                    d_csrRowPtr_coo2csr,
                    CUSPARSE_INDEX_BASE_ZERO ));

    CUDA_CALL(cudaMemcpy(h_A_COO.cooRowInd, d_csrRowPtr_coo2csr, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Input Data File Sample:\n");    
    printf("nnz = %d\n", h_A.nnz);
    for (int i=0;i<10;i++){
        printf("h_A.csrVal: %f\n",h_A.csrVal[i]);
        printf("h_A.csrRowPtr: %d\n",h_A.csrRowPtr[i]);
        printf("h_A.csrColInd: %d\n",h_A.csrColInd[i]);
    }

    printf("COO2CSR: Input Data File Sample:\n");    
    printf("nnz = %d\n", h_A_COO.nnz);
    for (int i=0;i<10;i++){
        printf("h_A_COO.cooVal: %f\n",h_A_COO.cooVal[i]);
        printf("h_A_COO.cooRowInd: %d\n",h_A_COO.cooRowInd[i]);
        printf("h_A_COO.cooColInd: %d\n",h_A_COO.cooColInd[i]);
    }

    int i;
    for(i=0;i<h_A.nnz;i++)
    {
        if ( (h_A.csrVal[i] != h_A_COO.cooVal[i]) || (h_A.csrColInd[i] != h_A_COO.cooColInd[i]) )
            printf("Col ERROR\n");

        if (i < N + 1)
            if ( h_A.csrRowPtr[i] != h_A_COO.cooRowInd[i])
                printf("Row ERROR\n");
    }
    printf("h_A.nnz = %d = %d = i\n", h_A.nnz, i);

    /* Cleanup */
    CUDA_CALL(cudaFree(d_csrRowPtr_coo2csr));
    CUDA_CALL(cudaFree(d_A_COO.cooVal));    CUDA_CALL(cudaFree(d_A_COO.cooRowInd));     CUDA_CALL(cudaFree(d_A_COO.cooColInd));
    free(h_A_COO.cooVal);                   free(h_A_COO.cooRowInd);                    free(h_A_COO.cooColInd);

    int baseB;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &(d_B.nnz);
    CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    // Allocate device memory to store the Row Pointers of the sparse CSR representation of B
    CUDA_CALL(cudaMalloc((void**)&(d_B.csrRowPtr), sizeof(int)*(N+1)));

    /* First determine the nnz of Sparse Matrix B */
    CHECK_CUSPARSE(cusparseXcsrgemmNnz(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        N, 
                        N, 
                        N,
                        descrA, 
                        d_A.nnz,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrA, 
                        d_A.nnz,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrB, 
                        d_B.csrRowPtr,
                        nnzTotalDevHostPtr ));

    if (NULL != nnzTotalDevHostPtr){
        d_B.nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CALL(cudaMemcpy(&(d_B.nnz), d_B.csrRowPtr+N, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&baseB,     d_B.csrRowPtr,   sizeof(int), cudaMemcpyDeviceToHost));
        d_B.nnz -= baseB;
    }

    /* Allocate device memory to store the rest of the sparse CSR representation of B */
    CUDA_CALL(cudaMalloc((void**)&d_B.csrVal,    sizeof(float) * d_B.nnz));
    CUDA_CALL(cudaMalloc((void**)&d_B.csrColInd, sizeof(int) * d_B.nnz));

                        /* Timer variable */
                        double first = cpuSecond();

    /* Perform the actual multiplication A * A = B */
    CHECK_CUSPARSE(cusparseScsrgemm(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        N, 
                        N, 
                        N,
                        descrA, 
                        d_A.nnz,
                        d_A.csrVal,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrA, 
                        d_A.nnz,
                        d_A.csrVal,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrB,
                        d_B.csrVal,
                        d_B.csrRowPtr, 
                        d_B.csrColInd ));

                        /* Timer display */
                        printf("GPU Sparse Matrices Multiplication wall clock time: %fs\n",cpuSecond()-first);


    /* Allocate memory onto the Host to hold the sparse CSR representation of B */
    h_B.nnz = d_B.nnz;
    h_B.csrVal = (float*)malloc ((h_B.nnz) * sizeof(float));
    h_B.csrRowPtr = (int*)malloc ((N + 1) * sizeof(int));
    h_B.csrColInd = (int*)malloc ((h_B.nnz) * sizeof(int));

    /* Copy the sparse CSR representation of B from the Device back to the Host */
    CUDA_CALL(cudaMemcpy(h_B.csrVal,    d_B.csrVal,    d_B.nnz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_B.csrRowPtr, d_B.csrRowPtr, (N + 1) * sizeof(int)  , cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_B.csrColInd, d_B.csrColInd, d_B.nnz * sizeof(int)  , cudaMemcpyDeviceToHost));


    printf("Input Data File Sample:\n");    
    printf("h_B.nnz = %d\n", d_B.nnz);
    for (int i=0;i<10;i++){
        printf("h_B.csrVal: %f\n",     h_B.csrVal[i]);
        printf("h_B.csrRowPtr: %d\n",  h_B.csrRowPtr[i]);
        printf("h_B.csrColInd: %d\n",  h_B.csrColInd[i]);
    }

    /* Cleanup */
    CUDA_CALL(cudaFree(d_A.csrVal));          CUDA_CALL(cudaFree(d_B.csrVal));
    CUDA_CALL(cudaFree(d_A.csrRowPtr));       CUDA_CALL(cudaFree(d_B.csrRowPtr));
    CUDA_CALL(cudaFree(d_A.csrColInd));       CUDA_CALL(cudaFree(d_B.csrColInd));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
