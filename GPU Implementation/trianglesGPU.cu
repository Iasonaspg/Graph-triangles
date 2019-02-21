/******************************************************************************
 *
 * trianglesGPU.cu -- The main function for the number of triangles (nT) finder 
 *                    algorithm implemented in CUDA
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <assert.h>
#include "readCSV.h"
#include "cuFindTriangles.h"
#include "validation.h"

int main (int argc, char **argv) {

  int *h_nT, nT_Mat, N, M,
      *d_nT;

  double matlab_time;

  csrFormat h_A, d_A; 

  /* Parsing input arguments */
  if ( argc == 2 ) 
    readCSV(argv[1], &h_A, &N, &M, &nT_Mat, &matlab_time);
  else {
    printf("Usage: ./trianglesGPU <CSVfileName>\n"); // <N> <M>
    printf(" where <CSVfileName.csv> is the name of the input data file (auto | great-britain_osm | delaunay_n22 | delaunay_n10)\n");
    printf("No need for suffix '.csv'\n");
    exit(1);
  }


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
  // 2D threads & blocks
  dim3 threads_per_block (warp/2, warp/2, 1);
  dim3 number_of_blocks ( SMs/4, SMs/4, 1);  


  /* Allocate device memory to store the sparse CSR representation of A */
  d_A.nnz = h_A.nnz;
  CUDA_CALL(cudaMalloc((void **)&(d_A.csrVal),    sizeof(float) * d_A.nnz));
  CUDA_CALL(cudaMalloc((void **)&(d_A.csrRowPtr), sizeof(int)   * (N + 1)));
  CUDA_CALL(cudaMalloc((void **)&(d_A.csrColInd), sizeof(int)   * d_A.nnz));

  /* Copy the sparse CSR representation of A from the Host to the Device */
  CUDA_CALL(cudaMemcpy(d_A.csrVal,    h_A.csrVal,    d_A.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_A.csrRowPtr, h_A.csrRowPtr, (N + 1) * sizeof(int)  , cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_A.csrColInd, h_A.csrColInd, d_A.nnz * sizeof(int)  , cudaMemcpyHostToDevice));

  /* Allocating memory to hold the nT variable (number of Triangles) */
  CUDA_CALL(cudaMalloc(&d_nT, 1 * sizeof(int)));
  h_nT = (int*)malloc (1 * sizeof(int));


  /* Zero out the content of the variable, 
  so that the summation result is valid */
  cuZeroVariable<<<1,1>>>( d_nT );

  /* Timer variables setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

            /* Begin timer */
            cudaEventRecord(start);
  
  // Calculating the Number of Triangles (nT) through the kernel 
  // (Only the sumation is performed here, the *(1/6) will be executed later on)
  cuFindTriangles<<<numberOfBlocks, threadsPerBlock>>>
    (d_A, N, d_nT);
  
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }

  CUDA_CALL(cudaDeviceSynchronize());

            /* Stop Timer */
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
  /* Copying nT, as calculated on GPU, back to the CPU */
  CUDA_CALL(cudaMemcpy(h_nT, d_nT, 1 * sizeof(int), cudaMemcpyDeviceToHost));

  /* Validating the result */
  int pass = validation(h_nT[0]/6, nT_Mat);    // Executing the nT*(1/6), that was omitted in cuFindTriangles
  assert(pass != 0);

  /* Calculate elapsed time */
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  /* Timer display */
  printf("GPU number of triangles nT: %d, Wall clock time: %fms ( < %lf ( Matlab Time ) )\n", h_nT[0], milliseconds, matlab_time);

            /* Write the results into file */
            FILE *fp;
            fp = fopen("GPU_Results.txt", "a");
            if ( fp == NULL ) {
              perror("Failed: Opening file Failed\n");
              return 1;
            }
            fprintf(fp, "%f\n", milliseconds);
            fclose(fp);


  /* CUDA Cleanup */
  CUDA_CALL(cudaFree(d_A.csrVal));
  CUDA_CALL(cudaFree(d_A.csrRowPtr));
  CUDA_CALL(cudaFree(d_A.csrColInd));

  /* Host Cleanup */
  free(h_A.csrVal);       free(h_A.csrRowPtr);      free(h_A.csrColInd);


  /* Exit */
  return 0;
}