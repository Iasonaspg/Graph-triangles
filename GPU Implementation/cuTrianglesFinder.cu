/**********************************************************************
 *
 * cuTrianglesFinder.cu -- Find the number of triangles in graph kernel
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuTrianglesFinder.h"

/* 
 * Can be probably accelerated further with some kind of implementation like the one shown in link below:
 * https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html 
 */
__global__
void cuTriangleFinder(int* C, int N, int *nT) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  for(int i = index; i < N * N; i += stride)
    atomicAdd( C[i], (*nT));

}
