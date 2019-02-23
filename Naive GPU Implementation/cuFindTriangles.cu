/******************************************************************************
 *
 * cuFindTriangles.cu -- The kernel that calculates the Number of triangles into
 *                       a graph given the CSR format of its Adjacency Matrix
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
void cuFindTriangles(csrFormat A, int N, int* nT) {

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int row = index; row < N; row += stride) {

      for (int j = A.csrRowPtr[row]; j < A.csrRowPtr[row+1]; j++) {

          int col = A.csrColInd[j];

          // [row, col] = position of 1 horizontally
          int beginPtr_csr_row = A.csrRowPtr[row];
          int beginPtr_csc_col = A.csrRowPtr[col];
          for (int k = beginPtr_csc_col; k < A.csrRowPtr[col+1]; k++) {
                  
              int csc_row = A.csrColInd[k];
              int csc_val = 1;
              // [csr_row, k] = position of 1 vertically

              for (int l = beginPtr_csr_row; l < A.csrRowPtr[row+1]; l++) {
    
                  int csr_col = A.csrColInd[l];
                  int csr_val = 1;

                  if ( csc_row == csr_col )
                      atomicAdd( nT, (int)(csr_val * csc_val) );
                  else if ( csr_col > csc_row ) {
                      beginPtr_csr_row = l;
                      break;
                  }

              }
          }
      }
  }

}


/*
// **************  2D threads & blocks  **************
{

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
// ***************************************************
*/
