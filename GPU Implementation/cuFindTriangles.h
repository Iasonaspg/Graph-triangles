/******************************************************************************
 *
 * cuFindTriangles.h -- The kernel that calculates the Number of triangles into
 *                       a graph given the CSR format of its Adjacency Matrix
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ******************************************************************************/

#ifndef CU_FIND_TRIANGLES_H
#define CU_FIND_TRIANGLES_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void cuZeroVariable(int* nT);

__global__
void cuFindTriangles(csrFormat A, int N, int* nT);

#endif /* CU_FIND_TRIANGLES_H */


#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#endif