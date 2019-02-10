/********************************************************************************
 *
 * trianglesGPU.cu -- The main function for the triangle finder algorithm in CUDA
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <assert.h>
#include "cuBandCFinder.h"
#include "cuTrianglesFinder.h"
#include "readMatFile.h"
#include "validation.h"

int main (int argc, char **argv) {

  int nT, *d_nT, nT_Mat;
  int *A, *d_A, *d_C, N, M;

  /* Parsing input arguments */
  if (argc > 1)
    N = 1<<atoi(argv[1]);  // Not sure if needed
    M = 1<<atoi(argv[2]);  // Not sure if needed
    nT_Mat = readMatFile(argv[3], A, &N);
  else{
    printf("Usage: ./triangles <matfile> <N> <M>\n");
    printf(" where <matfile> is the name of the MAT-file (auto | great-britain_osm | delaunay_n22)\n");
    printf(" where <N> is exp of number of Nodes in Graph\n");
    printf(" where <M> is exp of number of Edges in Graph\n");
    exit(1);
  }

  /* CUDA Device setup */
  size_t threadsPerBlock, warp;
  size_t numberOfBlocks, SMs;
  int deviceId;
  cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);
  warp = props.warpSize;
  SMs = props.SMsrocessorCount;
  threadsPerBlock = 8 * warp;
  numberOfBlocks  = 5 * SMs;

  dim3 threads_per_block (warp/2, warp/2, 1);
  dim3 number_of_blocks ( SMs/4, SMs/4, 1);  
    
  /* Timers setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocating memory to hold A matrix onto the GPU
  size_t ASize = N * N * sizeof(int);
  CUDA_CALL(cudaMalloc(&d_A,ASize));

  // Copying Input Data onto GPU
  CUDA_CALL(cudaMemcpy(A, d_A, ASize, cudaMemcpyHostToDevice));

  // Allocating memory to hold C matrix onto the GPU (same size as A)
  CUDA_CALL(cudaMalloc(&d_C,ASize));

  // Allocating memory to hold the nT variable (number of Triangles)
  CUDA_CALL(cudaMalloc(&nT, 1 * sizeof(int)));

            cudaEventRecord(start);
  
  // Calculating B matrix kernel (Normal Product)
  cuFindB<<<number_of_blocks, threads_per_block>>>
    (d_A, d_C, N);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }


  CUDA_CALL(cudaDeviceSynchronize());

  // Calculating C matrix kernel (Hadamard Product)
  cuFindC<<<numberOfBlocks, threadsPerBlock>>>
    (d_A, d_C, N);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }


  CUDA_CALL(cudaDeviceSynchronize());

  // Calculating the nT variable (number of Triangles) kernel 
  // (Only the sumation is performed here, the *(1/6) will be executed later on)
  cuTrianglesFinder<<<numberOfBlocks, threadsPerBlock>>>
    (d_C, N, d_nT);
  
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }


  CUDA_CALL(cudaDeviceSynchronize());

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
  
  // Copying nT, as calculated on GPU, back to the CPU
  CUDA_CALL(cudaMemcpy(nT, d_nT, 1 * sizeof(int), cudaMemcpyDeviceToHost));

  /* Validating the result */
  int pass = validation(nT/6, nT_Mat);    // Executing the nT*(1/6), that was omitted in cuTrianglesFinder
  assert(pass != 0);

  /* Print execution time */
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Implementation wall clock time: %1.6fms\n",milliseconds);

            /* File to keep the results */
            FILE *fp;
            fp = fopen("GPU_Results.txt", "a");
            if ( fp == NULL ) {
              perror("Failed: Opening file Failed\n");
              return 1;
            }
            fprintf(fp, "%f\n", cpu_time);
            fclose(fp);

  /* Cleanup */
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_C));
  CUDA_CALL(cudaFree(d_QBoxIdToCheck));
  free(A);
  
  /* Exit */
  return 0;
}