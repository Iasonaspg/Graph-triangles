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

    printf("Reading of dataset and validation file has started\n");
    readCSV(fName, &A, &N, &M, &nT_Mat, &matlab_time);
    printf("Reading of dataset and validation file has ended\n");

    printf("Nonzeros = %d\n", A.nnz);
 
    printf("Validation File:\n N = %d, M = %d\n Matlab result was %d, produced in %lf\n", N, M, nT_Mat, matlab_time);

    // Multiplication of two sparse arrays using cuSparse
    mulSparse(&A,&C,N);

    // Move adjancy array to gpu memory
    float* devVal;
    int* devCol, *devRow;
    int nnzA = A.nnz;
    CHECK(cudaMalloc((void**)&devVal,nnzA*sizeof(float)));
    CHECK(cudaMalloc((void**)&devCol,nnzA*sizeof(int)));
    CHECK(cudaMalloc((void**)&devRow,(N+1)*sizeof(int)));
    CHECK(cudaMemcpy(devVal,A.csrVal,nnzA*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devCol,A.csrColInd,nnzA*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devRow,A.csrRowPtr,(N+1)*sizeof(int),cudaMemcpyHostToDevice));

    // Use a struct to access it easier
    csrFormat B;
    B.csrColInd = devCol;
    B.csrRowPtr = devRow;
    B.csrVal = devVal;
    B.nnz = A.nnz;

    // Initialize value of triangles to 0
    int* sum, *h_sum;
    CHECK(cudaMalloc((void**)&sum,sizeof(int)));
    h_sum = (int*)malloc(sizeof(int));
    *h_sum = 0;
    CHECK(cudaMemcpy(sum,h_sum,sizeof(int),cudaMemcpyHostToDevice));
    
        
    // double st1 = cpuSecond();
    //filter<<<160,1024>>>(B,C,counter1,counter2);
    // CHECK(cudaPeekAtLastError());
    // CHECK(cudaDeviceSynchronize());
    //printf("Time filtering on GPU: %lf sec\n",cpuSecond()-st1);
    //printf("Vrhka tosa simeia: %d kai midenisa tosa: %d\n",*counter1,*counter2);

    // double st2 = cpuSecond();
    // findTriangles<<<160,1024>>>(B,C,sum,counter);
    // CHECK(cudaPeekAtLastError());
    // CHECK(cudaDeviceSynchronize());
    // printf("Time on GPU: %lf sec\n",cpuSecond()-st2);

    
    // Call our kernel to find the number of triangles
    double st3 = cpuSecond();
    findTriangles<<<20,1024>>>(B,C,sum,N);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_sum,sum,sizeof(int),cudaMemcpyDeviceToHost);
    printf("Triangles using CSR format: %d\n",h_sum[0]/3);
    //printf("Triangles naive: %d\n",counter[0]/6);
    printf("Time on GPU using CSR format: %lf sec\n",cpuSecond()-st3);

    if (validation(h_sum[0]/3,nT_Mat)){
        printf("Validation on GPU: PASSED\n");
    }
    

    // Bring matrix C back to host memory to call CPU function
    float* h_val = (float*)malloc(C.nnz*sizeof(float));
    int* h_col = (int*)malloc(C.nnz*sizeof(int));
    int* h_row = (int*)malloc((N+1)*sizeof(int));
    CHECK(cudaMemcpy(h_val,C.csrVal,C.nnz*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_col,C.csrColInd,C.nnz*sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_row,C.csrRowPtr,(N+1)*sizeof(int),cudaMemcpyDeviceToHost));

    // Pass it through a struct
    csrFormat D;
    D.csrColInd = h_col;
    D.csrRowPtr = h_row;
    D.csrVal = h_val;
    D.nnz = C.nnz;


    double st = cpuSecond();
    int triangles = findTrianglesCPU(&A,&D,N);
    printf("Time on CPU: %lf sec\n",cpuSecond()-st);

    if (validation(triangles,nT_Mat)){
        printf("Validation on CPU: PASSED\n");
    }
    
    
    free(h_sum);
    free(h_col);
    free(h_val);
    free(h_row);
    cudaFree(devCol);
    cudaFree(devRow);
    cudaFree(devVal);
    cudaFree(C.csrVal);
    cudaFree(C.csrColInd);
    cudaFree(C.csrRowPtr);
    cudaFree(sum);
    
    return 0; 
}