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
void cuTrianglesFinderHadamardOnly(csrFormat A, csrFormat B, int N, int* nT);

#endif /* CU_FIND_TRIANGLES_H */


#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#endif

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}
#endif    