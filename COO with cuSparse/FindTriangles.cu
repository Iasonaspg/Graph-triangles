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


__global__ void filter(cooFormat A, cooFormat C){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    

    for (int i=index;i<C.nnz;i+=stride){
        int flag = 0;
        for (int j=0;j<A.nnz;j++){
            if ((A.cooColIndA[j] == C.cooColIndA[i]) && (A.cooRowIndA[j] == C.cooRowIndA[i])){
                flag = 1;
                break;
            }
        }
        if (!flag){
            C.cooValA[i] = 0;
        }
    }

}


__global__ void findTriangles(cooFormat A, cooFormat C, int* sum, int* counter){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
     
    
    // for (int s=blockDim.x/2; s>0; s>>=1) {
    //     if (tid < s) {
    //         totalSum[tid] += totalSum[tid + s];
    //     }
    //     __syncthreads();
    // }

    if (threadIdx.x == 0 && blockIdx.x == 0){
        *sum = 0;
    }
    
    for (long i=index;i<C.nnz;i+=stride){
       for (int j=0;j<A.nnz;j++){
           if ((A.cooColIndA[j] == C.cooColIndA[i]) && (A.cooRowIndA[j] == C.cooRowIndA[i])){
               //atomicAdd(counter,1);
               atomicAdd(sum,C.cooValA[i]);
               break;
           }
       }
    //    __syncthreads();
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Triangles on GPU: %d\n",sum[0]/6);
        // printf("Mphka: %d\n",*counter);
    }
    
}

__global__ void findTrianglesShared(cooFormat A, cooFormat C, int* totalSum, int* counter){

    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    
    __shared__ int rowA[1024];
    __shared__ int colA[1024];
    

    if (threadIdx.x == 0 && blockIdx.x == 0){
        *totalSum = 0;
    }

    rowA[tid] = A.cooRowIndA[index];
    colA[tid] = A.cooColIndA[index];
    

    for (int i=0;i<C.nnz;i++){
        *counter = 0;
        if ((rowA[tid] == C.cooRowIndA[i]) && (colA[tid] == C.cooColIndA[i])){
            atomicAdd(totalSum,C.cooValA[i]);
            *counter = 1;
        }
        __syncthreads();
        if (*counter != 1){
            for (int j=(index+stride);j<A.nnz;j+=stride){
                if ((A.cooColIndA[j] == C.cooColIndA[i]) && (A.cooRowIndA[j] == C.cooRowIndA[i])){
                    //atomicAdd(counter,1);
                    atomicAdd(totalSum,C.cooValA[i]);
                    break;
                }
            }
        }
        __syncthreads();   
    }
        

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Triangles on GPU with shared memory: %d\n",totalSum[0]/6);
        // printf("Mphka: %d\n",*counter);
    }

}


void findTrianglesCPU(cooFormat* A, cooFormat* C){
    int sum = 0;
    for (int i=0;i<C->nnz;i++){
       for (int j=0;j<A->nnz;j++){
           if ((A->cooColIndA[j] == C->cooColIndA[i]) && (A->cooRowIndA[j] == C->cooRowIndA[i])){
               sum += C->cooValA[i];
               break;
           }
       }
    }
    printf("Triangles on CPU: %d\n",sum/6);
}