/**********************************************************************
 *
 * cuBandCFinder.h -- Calculate B and C matrixes from A kernels
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef CU_B_AND_C_FINDER_H
#define CU_B_AND_C_FINDER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void cuFindB(int* A, int* B, int N);

__global__
void cuFindC(int* A, int* C, int N);

#endif /* CU_B_AND_C_FINDER_H */


#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#endif