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
#include <cooperative_groups.h>
#include "readCSV.h"
#include "cuFindTriangles.h"

using namespace cooperative_groups;

__global__
/* Kernel function that zeros the number of triangles variable */
void cuZeroVariable(int* nT) {

  (*nT) = 0;
}


__global__
/* Kernel function that finds the number of triangles formed in the graph */
void cuTrianglesFinderHadamardOnly(csrFormat A, csrFormat B, int N, int* nT) {

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int row = index; row < N; row += stride) {

    int beginPtr_B = B.csrRowPtr[row];
    // Repeat into the row-th line range of A 
    for (int k = A.csrRowPtr[row]; k < A.csrRowPtr[row+1]; k++) {
      // When columns ( and lines by definition as well )  match,
      // then and only then keep the value ( Hadamard Product )
      int A_col = A.csrColInd[k];
      
      if ( A_col > row ) {
      // OPTIMIZATION: Due to symmetry, nT of the upper half array is
      // equal to half the nT, thus additions are cut down to half ! 
        
        // Repeat into the row-th line range of B  
        for (int j = beginPtr_B; j < B.csrRowPtr[row+1]; j++) {

          int B_col = B.csrColInd[j];

          if ( A_col == B_col )
              atomicAdd( nT, (int)( B.csrVal[j] ) );
          else if ( B_col > A_col ) {
              beginPtr_B = j;
              break;
          }
        }
      }
    }
  }
}

/*  Hadamard on CPU

    int nT = 0;

    // Repeat for every row
    for (int row = 0; row < N; row++) {

        int beginPtr_B = h_B.csrRowPtr[row];
        // Repeat into the row-th line range of A 
        for (int k = h_A.csrRowPtr[row]; k < h_A.csrRowPtr[row+1]; k++) {
            // When columns ( and lines by definition as well )  match,
            // then and only then keep the value ( Hadamard Product )
            int A_col = h_A.csrColInd[k];
            
            if ( A_col > row ) {
                // Repeat into the row-th line range of B  
                for (int j = beginPtr_B; j < h_B.csrRowPtr[row+1]; j++) {

                    int B_col = h_B.csrColInd[j];

                    if ( A_col == B_col )
                        nT += (int)( h_B.csrVal[j] );
                    else if ( B_col > A_col ) {
                        beginPtr_B = j;
                        break;
                    }
                }
            }
        }

    }
    printf("nT = %d\n", nT/3);

    validation(nT/3, nT_Mat);    
*/
