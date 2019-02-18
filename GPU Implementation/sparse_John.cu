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

__global__ void pri(float* tmp){
    printf("Random value: %f\n",tmp[4]);
}

int main(int argc, char** argv){

    int N, M, nT_Mat;
    double matlab_time;

    /* Create the struct of type csr Format to hold the Sparse Matrices A and B */
    csrFormat h_A, d_A, 
              h_B, d_B;

    /* Read the input Sparse Matrix, alongside with some further info */
    read_cSV(argv[1], &h_A, &N, &M, &nT_Mat, &matlab_time);

    /* Create the cuSPARSE handle */
    cusparseHandle_t handle = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));    

    /* Construct a descriptor of the matrix A */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    /* Construct a descriptor of the matrix B */
    cusparseMatDescr_t descrB = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate device memory to store the sparse CSR representation of A */
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrVal),    sizeof(float) * h_A.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrRowPtr), sizeof(int) * (N + 1)));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrColInd), sizeof(int) * h_A.nnz));

    int baseB;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &(h_B.nnz);
    CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    // Allocate device memory to store the Row Pointers of the sparse CSR representation of B
    CUDA_CALL(cudaMalloc((void**)&(d_B.csrRowPtr), sizeof(int)*(N+1)));

    /* First determine the nnz of Sparse Matrix B */
    CHECK_CUSPARSE(cusparseXcsrgemmNnz(cusparseHandle_t handle,
                        cusparseOperation_t CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        cusparseOperation_t CUSPARSE_OPERATION_NON_TRANSPOSE,
                        int N, 
                        int N, 
                        int N,
                        const cusparseMatDescr_t descrA, 
                        const int h_A.nnz,
                        const int d_A.csrRowPtr, 
                        const int d_A.csrColInd,
                        const cusparseMatDescr_t descrA, 
                        const int h_A.nnz,
                        const int d_A.csrRowPtr, 
                        const int d_A.csrColInd,
                        const cusparseMatDescr_t descrB, 
                        int d_B.csrRowPtr,
                        int nnzTotalDevHostPtr ));

    if (NULL != nnzTotalDevHostPtr){
        d_B.nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CALL(cudaMemcpy(&(d_B.nnz), d_B.csrRowPtr+N, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&baseB,     d_B.csrRowPtr, sizeof(int), cudaMemcpyDeviceToHost));
        d_B.nnz -= baseB;
    }

    /* Allocate device memory to store the rest of the sparse CSR representation of B */
    CUDA_CALL(cudaMalloc((void**)&d_B.csrColInd, sizeof(int) * d_B.nnz));
    CUDA_CALL(cudaMalloc((void**)&d_B.csrVal,    sizeof(float) * d_B.nnz));

    /* Perform the actual multiplication A * A = B */
    CHECK_CUSPARSE(cusparseScsrgemm(cusparseHandle_t handle,
                        cusparseOperation_t CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        cusparseOperation_t CUSPARSE_OPERATION_NON_TRANSPOSE,
                        int N, 
                        int N, 
                        int N,
                        const cusparseMatDescr_t descrA, 
                        const int h_A.nnz,
                        const float d_A.csrVal,
                        const int d_A.csrRowPtr, 
                        const int d_A.csrColInd,
                        const cusparseMatDescr_t descrA, 
                        const int h_A.nnz,
                        const float d_A.csrVal,
                        const int d_A.csrRowPtr, 
                        const int d_A.csrColInd,
                        const cusparseMatDescr_t descrB,
                        float d_B.csrVal,
                        const int d_B.csrRowPtr, 
                        int d_B.csrColInd ));

    /* Maybe transfer B matrix to the host and print some of the results */


    /* Cleanup */

    /***** Add any other cleanup here *****/

    CUDA_CALL(cudaFree(d_csrValA));          CUDA_CALL(cudaFree(d_csrValB));
    CUDA_CALL(cudaFree(d_csrRowPtrA));       CUDA_CALL(cudaFree(d_csrRowPtrB));
    CUDA_CALL(cudaFree(d_csrColIndA));       CUDA_CALL(cudaFree(d_csrColIndB));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
