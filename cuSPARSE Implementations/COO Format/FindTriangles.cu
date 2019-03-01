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


//  Kernel that calculates Hadamard product of our arrays
__global__ void filter(cooFormat A, cooFormat C, int* sum, int* counter2){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i=index;i<C.nnz;i+=stride){
        for (int j=0;j<A.nnz;j++){
            if ((C.cooColIndA[i] == A.cooColIndA[j]) && (C.cooRowIndA[i] == A.cooRowIndA[j])){
                atomicAdd(sum,C.cooValA[i]);
                break;
            }
        }
    }
}


// Kernel that calculates the number of triangles on a given matrix
__global__ void findTriangles(cooFormat A, cooFormat C, int* sum){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int index1 = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
     
    
    for (long i=index;i<C.nnz;i+=stride){
        if (C.cooColIndA[i] > C.cooRowIndA[i]){
            for (int j=0;j<A.nnz;j++){
                if ((A.cooColIndA[j] == C.cooColIndA[i]) && (A.cooRowIndA[j] == C.cooRowIndA[i])){
                     atomicAdd(sum,C.cooValA[i]);
                     break;
                }
            }
        }
    } 
}


// Kernel that sums the values of our array using reduction and shared memory
__global__ void findTrianglesSum(cooFormat A, cooFormat C, int* sum, int* counter){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x; 

    __shared__ int totalSum[1024];
    
    for (int i=index;i<C.nnz;i+=stride){
        totalSum[tid] = C.cooValA[index];
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
       
        // atomicAdd(counter,C.cooValA[i]);
        // atomicAdd(counter,1);
    }
}

// Function that finds the number of triangles
int findTrianglesCPU(cooFormat* A, cooFormat* C){
    int sum = 0;
    for (int i=0;i<A->nnz;i++){
        if (A->cooColIndA[i] > A->cooRowIndA[i]){
            for (int j=0;j<C->nnz;j++){
                if ((A->cooColIndA[i] == C->cooColIndA[j]) && (A->cooRowIndA[i] == C->cooRowIndA[j])){
                    sum += C->cooValA[j];
                    break;
                }
            }    
        }
    }
    printf("Triangles on CPU: %d\n",sum/3);
    return sum/3;
}