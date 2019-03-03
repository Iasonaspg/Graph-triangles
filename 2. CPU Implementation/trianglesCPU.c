/************************************************************************************
 *
 * trianglesCPU.c -- The CPU implementation for the number of triangles (nT) 
 *                   finder algorithm
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "readCSV.h"
#include "validation.h"

/* Forward declaration of the nT Calculator Function */
int findTriangles(csrFormat* A, int N);

/* Timer Function */
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main (int argc, char **argv) {

  int nT, nT_Mat, N, M;
  double matlab_time;

  csrFormat A;

  /* Parsing input arguments */
  if ( argc == 2 ) {
    printf("--Reading Input Data from CSV file: Started--\n");    
    readCSV(argv[1], &A, &N, &M, &nT_Mat, &matlab_time);
    printf("--Reading Input Data from CSV file: DONE!--\n");    
  } else {
    printf("Usage: ./trianglesCPU <CSVfileName>\n");
    printf(" where <CSVfileName>.csv is the name of the input data file (auto | great-britain_osm | delaunay_n22 | delaunay_n10)\n");
    printf("No need for suffix '.csv'\n");
    exit(1);
  }

                        /* Timer variable */
                        double timer = cpuSecond();
  nT = findTriangles(&A, N);
                        /* Calculate elapsed time */
                        timer = cpuSecond()-timer;


  /* Validating the result */
  int pass = validation(nT, nT_Mat);
  assert(pass != 0);

  /* Timer display */
  printf("CPU number of triangles nT: %d, Wall clock time: %lfs ( < %lf ( Matlab Time ) )\n", nT, timer, matlab_time);

            /* Write the results into file */
            FILE *fp;
            fp = fopen("CPU_Results.txt", "a");
            if ( fp == NULL ) {
              perror("Failed: Opening file Failed\n");
              return 1;
            }
            fprintf(fp, "%lf\n", timer);
            fclose(fp);

  /* Cleanup */
  free(A.csrRowPtr);      free(A.csrColInd);
  
  /* Exit */
  return 0;
}


/* Function that finds the number of triangles formed in the graph */
int findTriangles(csrFormat* A, int N)
{
  int nT = 0;

  // Iterate over rows
  for (int row = 0; row < N; row++) {
    // Iterate over columns
    for (int j = A->csrRowPtr[row]; j < A->csrRowPtr[row+1]; j++) {

        int col = A->csrColInd[j];
        // [row, col] = position of 1 horizontally
        
        if ( col>row ) {
        // OPTIMIZATION: Due to symmetry, nT of the upper half array is
        // equal to half the nT, thus additions are cut down to half ! 
          int beginPtr_csr_row = A->csrRowPtr[row];
          int beginPtr_csc_col = A->csrRowPtr[col];
          
          // Multiplication of A[:,col] * A[row,:]      
          for (int k = beginPtr_csc_col; k < A->csrRowPtr[col+1]; k++) {
            int csc_row = A->csrColInd[k];
            // [csr_row, k] = position of 1 vertically

            for (int l = beginPtr_csr_row; l < A->csrRowPtr[row+1]; l++) {

                int csr_col = A->csrColInd[l];

                if ( csc_row == csr_col )
                    nT++; 
                else if ( csr_col > csc_row ) {
                // OPTIMIZATION: when col>row no need to go further,
                // continue to the next col, plus for further optimization
                // keep track of the beginPtr_csr_row where the previous
                // iteration stopped, so that no time is wasted in rechecking
                    beginPtr_csr_row = l;
                    break;
                }

            }
          }
        }
    }
  }

  // nT = (1/6) * nT;
  return nT/3;
  // though as of condition if ( col>row ) we have cut additions to half
  // due to the symmetry of the adjacency matrix, so 2*(nT/6) = nT/3
}
