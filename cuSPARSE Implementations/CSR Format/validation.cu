/**********************************************************************
 *
 * validation.cu (.c) -- Validation function for the number of triangles 
 *                    	 calculated
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusparse_v2.h>
#include "validation.h"
#include "cuFindTriangles.h"

int validation( int nT, int nT_Mat )
{

  if ( nT != nT_Mat )
  {
    printf("Validation FAILED: nT = %d, while correct value nT_Mat = %d \n", nT, nT_Mat);
    return 0;
  }

  return 1;
}


__global__ void cuValidateCSR( csrFormat A_CSR, cooFormat A_coo2csr, int nnz, int N, int* valid)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
  	int stride = blockDim.x * gridDim.x;
	
	for(int i = index; i < nnz; i += stride)
	{
	    if ( (A_CSR.csrVal[i] != A_coo2csr.cooVal[i]) || (A_CSR.csrColInd[i] != A_coo2csr.cooColInd[i]) ) {
	        printf("Col ERROR\n");
	    	(*valid) = 1;
	    }

	    if ( i < N + 1 )
	        if ( A_CSR.csrRowPtr[i] != A_coo2csr.cooRowInd[i]) {
	            printf("Row ERROR\n");
	            (*valid) = 1;
	        }
	}
}

int validateCSR( cusparseHandle_t handle, csrFormat d_A_CSR, cooFormat h_A_COO, int N )
{
	int *h_valid, 
		*d_valid;

    cooFormat d_A_coo2csr; 

    int* d_cooRowInd2csrRowPtr;

    /* CUDA Device setup */
    size_t threadsPerBlock, warp;
    size_t numberOfBlocks, SMs;
    cudaError_t err;
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    warp = props.warpSize;
    SMs = props.multiProcessorCount;
    // 1D threads & blocks
    threadsPerBlock = 8 * warp;
    numberOfBlocks  = 5 * SMs;

    /* Construct a descriptor of the matrix A_COO */
    cusparseMatDescr_t descrA_COO = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA_COO));
    CHECK_CUSPARSE(cusparseSetMatType(descrA_COO, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA_COO, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate device memory to store the sparse COO representation of A */
    d_A_coo2csr.nnz = h_A_COO.nnz;
    CUDA_CALL(cudaMalloc((void **)&(d_A_coo2csr.cooVal),    sizeof(float) * d_A_coo2csr.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_A_coo2csr.cooRowInd), sizeof(int) * d_A_coo2csr.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_cooRowInd2csrRowPtr), sizeof(int) * (N + 1)));
    CUDA_CALL(cudaMalloc((void **)&(d_A_coo2csr.cooColInd), sizeof(int) * d_A_coo2csr.nnz));

    /* Copy the sparse COO representation of A from the Host to the Device */
    CUDA_CALL(cudaMemcpy(d_A_coo2csr.cooVal,    h_A_COO.cooVal,    d_A_coo2csr.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A_coo2csr.cooRowInd, h_A_COO.cooRowInd, d_A_coo2csr.nnz * sizeof(int)  , cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A_coo2csr.cooColInd, h_A_COO.cooColInd, d_A_coo2csr.nnz * sizeof(int)  , cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseXcoo2csr(handle,
                    d_A_coo2csr.cooRowInd,
                    d_A_coo2csr.nnz,
                    N,
                    d_cooRowInd2csrRowPtr,
                    CUSPARSE_INDEX_BASE_ZERO ));

	CUDA_CALL(cudaFree(d_A_coo2csr.cooRowInd));     
    d_A_coo2csr.cooRowInd = d_cooRowInd2csrRowPtr;

    /* Allocating memory to hold the valid variable (number of Triangles) */
    CUDA_CALL(cudaMalloc(&d_valid, 1 * sizeof(int)));
    h_valid = (int*)malloc (1 * sizeof(int));

    /* Zero out the content of the variable, 
    so that the summation result is valid */
    cuZeroVariable<<<1,1>>>( d_valid );

    CUDA_CALL(cudaDeviceSynchronize());

    cuValidateCSR<<<numberOfBlocks, threadsPerBlock>>>
    (d_A_CSR, d_A_coo2csr, d_A_CSR.nnz, N, d_valid);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }

    /* Copying valid back to the CPU */
    CUDA_CALL(cudaMemcpy(h_valid, d_valid, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    /* Cleanup */
    CUDA_CALL(cudaFree(d_cooRowInd2csrRowPtr));
    CUDA_CALL(cudaFree(d_A_coo2csr.cooVal));    CUDA_CALL(cudaFree(d_A_coo2csr.cooColInd));
    free(h_A_COO.cooVal);                   free(h_A_COO.cooRowInd);                    free(h_A_COO.cooColInd);

    return !(*h_valid);
}