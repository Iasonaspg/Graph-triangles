/************************************************************************************
 *
 * trianglesCPU.cu (.c) -- The CPU implementation of the triangle finder algorithm
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
#include "readMatFile.h"
#include "validation.h"

int findTriangles(int *A, int N);

int main (int argc, char **argv) {

  int nT, nT_Mat;
  int *A, N, M;

  /* Variables to hold execution time */
  struct timeval startwtime, endwtime;
  double cpu_time;

  /* Parsing input arguments */
  if (argc > 1) {
    N = 1<<atoi(argv[1]);  // Not sure if needed
    M = 1<<atoi(argv[2]);  // Not sure if needed
    nT_Mat = readMatFile(argv[3], A, &N);
  } else{
    printf("Usage: ./triangles <matfile> <N> <M>\n");
    printf(" where <matfile> is the name of the MAT-file (auto | great-britain_osm | delaunay_n22)\n");
    printf(" where <N> is exp of number of Nodes in Graph\n");
    printf(" where <M> is exp of number of Edges in Graph\n");
    exit(1);
  }

            gettimeofday (&startwtime, NULL);
  nT = findTriangles(A, N);
            gettimeofday (&endwtime, NULL);


  /* Validating the result */
  int pass = validation(nT, nT_Mat);
  assert(pass != 0);

  cpu_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                      + endwtime.tv_sec - startwtime.tv_sec);

  /* Print execution time */
  printf("CPU Implementation wall clock time: %f sec\n", cpu_time);

            /* File to keep the results */
            FILE *fp;
            fp = fopen("CPU_Results.txt", "a");
            if ( fp == NULL ) {
              perror("Failed: Opening file Failed\n");
              return 1;
            }
            fprintf(fp, "%f\n", cpu_time);
            fclose(fp);

  /* Cleanup */
  free(A);
  
  /* Exit */
  return 0;
}


/* Function that finds the number of triangles formed in the graph */
int findTriangles(int *A, int N)
{

  int *C;
  int nT = 0;

  // Allocating memory for the C matrix
  C = (int *) malloc( N * N * sizeof(int) );

  // C = A * A; (Normal Product)
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      if (A[row * N + col] != 0)
      {
        val = 0;
        for ( int k = 0; k < N; ++k )
          val += A[row * N + k] * A[k * N + col];
        C[row * N + col] = val;
      }
    }

  // C = C o A; (Hadamard Product) 
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
      C[row * N + col] = C[row * N + col] * A[row * N + col];

  // nT = sum(C);
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
      nT += C[row * N + col];

  // nT = (1/6) * nT;
  return nT/6;
}
