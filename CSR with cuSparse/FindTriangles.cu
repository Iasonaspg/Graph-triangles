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


__global__ void filter(csrFormat A, csrFormat C, int* counter1, int* counter2){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    if (threadIdx.x == 0 && blockIdx.x == 0){
        *counter1 = 0;
        *counter2 = 0;
    }

    for (int i=index;i<C.nnz;i+=stride){
        int flag = 0;
        for (int j=0;j<A.nnz;j++){
            if ((A.csrColInd[j] == C.csrColInd[i]) && (A.csrRowPtr[j] == C.csrRowPtr[i])){
                flag = 1;
                //atomicAdd(counter1,1);
                break;
            }
        }
        if (flag == 0){
            C.csrVal[i] = 0;
            //atomicAdd(counter2,1);
        }
    }
}


__global__ void findTriangles(csrFormat A, csrFormat C, int* sum, int N){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
     
   
    for (int i=index;i<C.nnz;i+=stride){
        for (int j=0;j<N;j++){
            for (int k=A.csrRowPtr[j];k<A.csrRowPtr[j+1];k++){
                if ((A.csrColInd[k] == C.csrColInd[i]) && (i >= C.csrRowPtr[j]) && (i < C.csrRowPtr[j+1])){
                    atomicAdd(sum,C.csrVal[i]);
                    j = N;
                    break;
                }
            }
           
        }
    }
}


__global__ void findTrianglesSum(csrFormat A, csrFormat C, int* sum, int* counter){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x; 

    __shared__ int totalSum[1024];
    
    for (int i=index;i<C.nnz;i+=stride){
        totalSum[tid] = C.csrVal[index];
        __syncthreads();

        for (int s=blockDim.x/2; s>0; s>>=1) {
            if (tid < s) {
                totalSum[tid] += totalSum[tid + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0){
            atomicAdd(sum,totalSum[0]);
            atomicAdd(counter,1);
        }
        __syncthreads();
    }

    

    for (long i=index;i<C.nnz;i+=stride){
       
        // atomicAdd(counter,C.csrValA[i]);
        // atomicAdd(counter,1);
    }
}

void findTrianglesCPU(csrFormat* A, csrFormat* C, int N){
    int sum = 0;
    for (int i=0;i<A->nnz;i++){
        for (int j=0;j<N;j++){
            for (int k=C->csrRowPtr[j];k<C->csrRowPtr[j+1];k++){
                if ((C->csrColInd[k] == A->csrColInd[i]) && (i >= A->csrRowPtr[j]) && (i < A->csrRowPtr[j+1])){
                    sum += C->csrVal[k];
                    j = N;
                    break;
                }
            }
           
        }
    }
    printf("Triangles on CPU: %d\n",sum/6);
}