/******************************************************************************
 *
 * cuFindTriangles.cu -- The kernel that calculates the Number of triangles into
 *                       a graph given the CSR format of its Adjacency Matrix
 *
 * Reference: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 * 
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "readCSV.h"
#include "cuFindTriangles.h"

__global__
/* Kernel function that zeros the number of triangles variable */
void cuZeroVariable(int* nT) {

  (*nT) = 0;
}


__global__
/* Kernel function that finds the number of triangles formed in the graph */
void cuTrianglesFinderHadamardOnly(cooFormat A, cooFormat B, int A_nnz, int B_nnz, int* nT){

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  // int begin_B = 0;
  for (int i = index; i < A_nnz; i += stride){
      if (A.cooColInd[i] > A.cooRowInd[i]){
          for (int j = 0; j < B_nnz; j++){
              if ((A.cooColInd[i] == B.cooColInd[j]) && (A.cooRowInd[i] == B.cooRowInd[j])){
                atomicAdd( nT, B.cooVal[j] );
                // begin_B = j + 1;
                break;
              }
          }
      }
  } 

}
