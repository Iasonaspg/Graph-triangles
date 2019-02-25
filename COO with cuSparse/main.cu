/**********************************************************************
 *
 * readCSV.c -- readCSV function for reading the input from
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <ctype.h>  
 #include <inttypes.h>
 #include <errno.h>
 #include <math.h>
 #include "readCSV.h"

 

int main(int argc, char** argv){
 
    char* fName = argv[1];
    int N, M, nT_Mat;
    double matlab_time;

    cooFormat A, C;

    readCSV(fName, &A, &N, &M, &nT_Mat, &matlab_time);

    
    printf("Nonzeros = %d\n", A.nnz);
    

    printf("Validation File:\n N = %d, M = %d\n Matlab result was %d, produced in %lf\n", N, M, nT_Mat, matlab_time);

    mulSparse(&A,&C,N);

    float* devVal;
    int* devCol, *devRow;
    int nnzA = A.nnz;
    cudaMallocManaged(&devVal,nnzA*sizeof(float));
    cudaMallocManaged(&devCol,nnzA*sizeof(int));
    cudaMallocManaged(&devRow,nnzA*sizeof(int));
    cudaMemcpy(devVal,A.cooValA,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A.cooColIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRow,A.cooRowIndA,nnzA*sizeof(int),cudaMemcpyHostToDevice);

    cooFormat B;
    B.cooColIndA = devCol;
    B.cooRowIndA = devRow;
    B.cooValA = devVal;
    B.nnz = A.nnz;

    int* sum, *counter;
    cudaMalloc((void**)&sum,sizeof(int));
    cudaMalloc((void**)&counter,sizeof(int));

    double st1 = cpuSecond();
    filter<<<160,1024>>>(B,C);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    printf("Time filtering on GPU: %lf sec\n",cpuSecond()-st1);

    double st2 = cpuSecond();
    findTrianglesShared<<<160,1024>>>(B,C,sum,counter);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    printf("Time on GPU using shared memory: %lf sec\n",cpuSecond()-st2);

    double st3 = cpuSecond();
    findTriangles<<<160,1024>>>(B,C,sum,counter);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    printf("Time on GPU using memory: %lf sec\n",cpuSecond()-st3);
    
    double st = cpuSecond();
    //findTrianglesCPU(&B,&C);
    printf("Time on CPU: %lf sec\n",cpuSecond()-st);

    // for (int i=0;i<9;i++){
    // printf("Sample: %f\n",C.cooValA[i]);
    // }
    
    return 0; 
}