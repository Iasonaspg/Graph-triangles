/******************************************************************************
 *
 * cuFindTriangles.cu -- The kernel that calculates the Number of triangles into
 *                       a graph given the CSR format of its Adjacency Matrix
 *
 * Reference: 
 *    https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 * 
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include "readCSV.h"
#include "cuFindTriangles.h"

using namespace cooperative_groups;

__global__
/* Kernel function that zeros the number of triangles variable */
void cuZeroVariable(int* nT) {

  (*nT) = 0;
}


__device__ void atomicAggInc(int *ctr) {
  auto g = coalesced_threads();
  if(g.thread_rank() == 0)
    atomicAdd(ctr, g.size());
  return ;
}


__global__
/* Kernel function that finds the number of triangles formed in the graph */
void cuFindTriangles(csrFormat A, int N, int* nT) {

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int row = index; row < N; row += stride) {

      for (int j = A.csrRowPtr[row]; j < A.csrRowPtr[row+1]; j++) {

          int col = A.csrColInd[j];
          // [row, col] = position of 1 horizontally

          if ( col>row ) {
            int beginPtr_csr_row = A.csrRowPtr[row];
            int beginPtr_csc_col = A.csrRowPtr[col];
            for (int k = beginPtr_csc_col; k < A.csrRowPtr[col+1]; k++) {
                    
                int csc_row = A.csrColInd[k];
                // [csr_row, k] = position of 1 vertically

                for (int l = beginPtr_csr_row; l < A.csrRowPtr[row+1]; l++) {
      
                    int csr_col = A.csrColInd[l];

                    if ( csc_row == csr_col )
                        atomicAggInc( nT );
                    else if ( csr_col > csc_row ) {
                        beginPtr_csr_row = l;
                        break;
                    }

                }
            }
          }
      }
  }

}
