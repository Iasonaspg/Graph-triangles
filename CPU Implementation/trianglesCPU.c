/************************************************************************************
 *
 * trianglesCPU.cu (.c) -- The CPU implementation of the triangle finder algorithm
 *
 * Reference: https://www.geeksforgeeks.org/operations-sparse-matrices/
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

int findTriangles(int *A, int N);

int main (int argc, char **argv) {

  int nT, nT_Mat;
  int *A, N, M;

  /* Variables to hold execution time */
  struct timeval startwtime, endwtime;
  double cpu_time, matlab_time;

  /* Parsing input arguments */
  if (argc == 4) {
    // N = 1<<atoi(argv[1]);  // Not sure if needed
    // M = 1<<atoi(argv[2]);  // Not sure if needed
    readCSV(argv[1], A, &N, &nT_Mat, &matlab_time);
  } else {
    printf("Usage: ./triangles <CSVfileName>\n"); // <N> <M>
    // printf(" where <N> is exp of number of Nodes in Graph\n");
    // printf(" where <M> is exp of number of Edges in Graph\n");
    printf(" where <CSVfileName.csv> is the name of the input data file (auto | great-britain_osm | delaunay_n22)\n");
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

  // int *C;
  int nT = 0;

  // Allocating memory for the C matrix
  // C = (int *) malloc( N * N * sizeof(int) );

  

  // nT = (1/6) * nT;
  return nT/6;
}

  public void add(sparse_matrix b) 
  { 

      int apos = 0, bpos = 0; 
      sparse_matrix result = new sparse_matrix(row, col); 

      while (apos < len && bpos < b.len) { 

        // if b's row and col is smaller 
        if (data[apos,0] > b.data[bpos,0] || 
        (data[apos,0] == b.data[bpos,0] && 
        data[apos,1] > b.data[bpos,1])) 

        { 

          // insert smaller value into result 
          result.insert(b.data[bpos,0], 
                b.data[bpos,1], 
                b.data[bpos,2]); 

          bpos++; 
        } 

        // if a's row and col is smaller 
        else if (data[apos,0] < b.data[bpos,0] || 
        (data[apos,0] == b.data[bpos,0] && 
        data[apos,1] < b.data[bpos,1])) 

        { 

          // insert smaller value into result 
          result.insert(data[apos,0], 
                data[apos,1], 
                data[apos,2]); 

          apos++; 
        } 

        else { 

          // add the values as row and col is same 
          int addedval = data[apos,2] + b.data[bpos,2]; 

          if (addedval != 0) 
            result.insert(data[apos,0], 
                  data[apos,1], 
                  addedval); 
          // then insert 
          apos++; 
          bpos++; 
        } 
      } 

      // insert remaining elements 
      while (apos < len) 
        result.insert(data[apos,0], 
              data[apos,1], 
              data[apos++,2]); 

      while (bpos < b.len) 
        result.insert(b.data[bpos,0], 
              b.data[bpos,1], 
              b.data[bpos++,2]); 

      // print result 
      result.print(); 
  } 

  public sparse_matrix transpose() 
  { 

    // new matrix with inversed row X col 
    sparse_matrix result = new sparse_matrix(col, row); 

    // same number of elements 
    result.len = len; 

    // to count number of elements in each column 
    int[] count = new int[col + 1]; 

    // initialize all to 0 
    for (int i = 1; i <= col; i++) 
      count[i] = 0; 

    for (int i = 0; i < len; i++) 
      count[data[i,1]]++; 

    int[] index = new int[col + 1]; 

    // to count number of elements having col smaller 
    // than particular i 

    // as there is no col with value < 1 
    index[1] = 0; 

    // initialize rest of the indices 
    for (int i = 2; i <= col; i++) 

      index[i] = index[i - 1] + count[i - 1]; 

    for (int i = 0; i < len; i++) { 

      // insert a data at rpos and increment its value 
      int rpos = index[data[i,1]]++; 

      // transpose row=col 
      result.data[rpos,0] = data[i,1]; 

      // transpose col=row 
      result.data[rpos,1] = data[i,0]; 

      // same value 
      result.data[rpos,2] = data[i,2]; 
    } 

    // the above method ensures 
    // sorting of transpose matrix 
    // according to row-col value 
    return result; 
  } 

  public void multiply(sparse_matrix b) 
  { 

    if (col != b.row) { 

      // Invalid muliplication 
      System.Console.WriteLine("Can't multiply, "
              + "Invalid dimensions"); 

      return; 
    } 

    // transpose b to compare row 
    // and col values and to add them at the end 
    b = b.transpose(); 
    int apos, bpos; 

    // result matrix of dimension row X b.col 
    // however b has been transposed, hence row X b.row 
    sparse_matrix result = new sparse_matrix(row, b.row); 

    // iterate over all elements of A 
    for (apos = 0; apos < len;) { 

      // current row of result matrix 
      int r = data[apos,0]; 

      // iterate over all elements of B 
      for (bpos = 0; bpos < b.len;) { 

        // current column of result matrix 
        // data[,0] used as b is transposed 
        int c = b.data[bpos,0]; 

        // temporary pointers created to add all 
        // multiplied values to obtain current 
        // element of result matrix 
        int tempa = apos; 
        int tempb = bpos; 

        int sum = 0; 

        // iterate over all elements with 
        // same row and col value 
        // to calculate result[r] 
        while (tempa < len && data[tempa,0] == r 
          && tempb < b.len && b.data[tempb,0] == c) { 

          if (data[tempa,1] < b.data[tempb,1]) 

            // skip a 
            tempa++; 

          else if (data[tempa,1] > b.data[tempb,1]) 

            // skip b 
            tempb++; 
          else

            // same col, so multiply and increment 
            sum += data[tempa++,2] * b.data[tempb++,2]; 
        } 

        // insert sum obtained in result[r] 
        // if its not equal to 0 
        if (sum != 0) 
          result.insert(r, c, sum); 

        while (bpos < b.len && b.data[bpos,0] == c) 

          // jump to next column 
          bpos++; 
      } 

      while (apos < len && data[apos,0] == r) 

        // jump to next row 
        apos++; 
    } 

    result.print(); 
  } 

  public static void Main() 
  { 

    // create two sparse matrices and insert values 
    sparse_matrix a = new sparse_matrix(4, 4); 
    sparse_matrix b = new sparse_matrix(4, 4); 

    a.insert(1, 2, 10); 
    a.insert(1, 4, 12); 
    a.insert(3, 3, 5); 
    a.insert(4, 1, 15); 
    a.insert(4, 2, 12); 
    b.insert(1, 3, 8); 
    b.insert(2, 4, 23); 
    b.insert(3, 3, 9); 
    b.insert(4, 1, 20); 
    b.insert(4, 2, 25); 

    // Output result 
    System.Console.WriteLine("Addition: "); 
    a.add(b); 
    System.Console.WriteLine("\nMultiplication: "); 
    a.multiply(b); 
    System.Console.WriteLine("\nTranspose: "); 
    sparse_matrix atranspose = a.transpose(); 
    atranspose.print(); 
  }