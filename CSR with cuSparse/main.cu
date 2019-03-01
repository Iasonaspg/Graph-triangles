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
 #include "validation.h"

 

int main(int argc, char** argv){
 
    char* fName = argv[1];
    int N, M, nT_Mat;
    double matlab_time;

    csrFormat A, C;

    readCSV(fName, &A, &N, &M, &nT_Mat, &matlab_time);

    
    printf("Nonzeros = %d\n", A.nnz);
    

    printf("Validation File:\n N = %d, M = %d\n Matlab result was %d, produced in %lf\n", N, M, nT_Mat, matlab_time);

    mulSparse(&A,&C,N);

    float* devVal;
    int* devCol, *devRow;
    int nnzA = A.nnz;
    cudaMallocManaged(&devVal,nnzA*sizeof(float));
    cudaMallocManaged(&devCol,nnzA*sizeof(int));
    cudaMallocManaged(&devRow,(N+1)*sizeof(int));
    cudaMemcpy(devVal,A.csrVal,nnzA*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devCol,A.csrColInd,nnzA*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(devRow,A.csrRowPtr,(N+1)*sizeof(int),cudaMemcpyHostToDevice);

    csrFormat B;
    B.csrColInd = devCol;
    B.csrRowPtr = devRow;
    B.csrVal = devVal;
    B.nnz = A.nnz;

    int* sum, *counter, *counter1, *counter2;
    cudaMallocManaged(&sum,sizeof(int));
    cudaMallocManaged((void**)&counter,sizeof(int));
    cudaMallocManaged(&counter1,sizeof(int));
    cudaMallocManaged(&counter2,sizeof(int));

    double st1 = cpuSecond();
    //filter<<<160,1024>>>(B,C,counter1,counter2);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    //printf("Time filtering on GPU: %lf sec\n",cpuSecond()-st1);
    //printf("Vrhka tosa simeia: %d kai midenisa tosa: %d\n",*counter1,*counter2);

    double st2 = cpuSecond();
    // findTriangles<<<160,1024>>>(B,C,sum,counter);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    // printf("Time on GPU: %lf sec\n",cpuSecond()-st2);

    double st3 = cpuSecond();
    *sum = 0;
    findTriangles<<<160,1024>>>(B,C,sum,N);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    printf("Triangles using CSR format: %d\n",sum[0]/3);
    //printf("Triangles naive: %d\n",counter[0]/6);
    printf("Time on GPU using CSR format: %lf sec\n",cpuSecond()-st3);

    if (validation(sum[0]/3,nT_Mat)){
        printf("Validation on GPU: PASSED\n");
    }
    
    double st = cpuSecond();
    // findTrianglesCPU(&B,&C,N);
    // printf("Time on CPU: %lf sec\n",cpuSecond()-st);
    
    return 0; 
}