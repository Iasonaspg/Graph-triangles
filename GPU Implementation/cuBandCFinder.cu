/**********************************************************************
 *
 * cuBandCFinder.cu -- Calculate B and C matrixes from A kernels
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuBandCFinder.h"

__global__
void cuFindB(int* A, int* B, int N) {

  int val = 0;

  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  int col_index = blockIdx.y * blockDim.y + threadIdx.y;
  int row_stride = blockDim.x * gridDim.x;
  int col_stride = blockDim.y * gridDim.y;

  for(int row = row_index; row < N; row += row_stride)
  {
    for(int col = col_index; col < N; col += col_stride)
      if ( (A[row * N + col] != 0) )
      {
        val = 0;
        for ( int k = 0; k < N; ++k )
          val += A[row * N + k] * A[k * N + col];
        B[row * N + col] = val;
      }
  }
}


__global__
void cuFindC(int* A, int* C, int N) {
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
    C[i] = B[i] * A[i];

}
