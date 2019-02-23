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
 
 
__global__ void findTriangles(cooFormat A, cooFormat C, int* sum, int* counter){
    long index = threadIdx.x + blockIdx.x*blockDim.x;
    int tid = threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    //printf("To A: %d\n",A.nnz);
   
    *sum = 0;
    // __shared__ float totalSum[1024];
    // totalSum[tid] = C.cooValA[index];
    // if (threadIdx.x == 0){
    //     for (int k=0;k<9;k++){
    //         printf("Times: %")
    //     }
    // }
    // __syncthreads();
    
    // for (int s=blockDim.x/2; s>0; s>>=1) {
    //     if (tid < s) {
    //         totalSum[tid] += totalSum[tid + s];
    //     }
    //     __syncthreads();
    // }

    
    for (long i=index;i<C.nnz;i+=stride){
       for (int j=0;j<A.nnz;j++){
           if ((A.cooColIndA[j] == C.cooColIndA[i]) && (A.cooRowIndA[j] == C.cooRowIndA[i])){
               atomicAdd(counter,1);
               atomicAdd(sum,C.cooValA[i]);
               break;
           }
       }
       __syncthreads();
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("Triangles on GPU: %d\n",sum[0]/6);
        printf("Mphka: %d\n",*counter);
    }
    
 }

 void findTrianglesCPU(cooFormat* A, cooFormat* C){
    int sum = 0;
    for (int i=0;i<C->nnz;i++){
       for (int j=0;j<A->nnz;j++){
           if ((A->cooColIndA[j] == C->cooColIndA[i]) && (A->cooRowIndA[j] == C->cooRowIndA[i])){
               //atomicAdd(&sum,C.cooValA[i]);
               sum += C->cooValA[i];
               break;
           }
       }
    }
    printf("Triangles on CPU: %d\n",sum/6);
 }
 
int main(int argc, char** argv){
 
    char* fName = argv[1];
    int N, M, nT_Mat;
    double matlab_time;

    cooFormat A, C;

    readCSV(fName, &A, &N, &M, &nT_Mat, &matlab_time);

    printf("Input Data File Sample:\n");    
    printf("nnz = %d\n", A.nnz);
    for (int i=1000;i<1012;i++){
        //printf("cooRowIndA: %d\n",A.cooRowIndA[i]);
        //printf("cooColIndA: %d\n",A.cooColIndA[i]);
        //printf("cooValA: %f\n",A.cooValA[i]);
    }

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
    findTriangles<<<2,1024>>>(B,C,sum,counter);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    printf("Time on GPU: %lf sec\n",cpuSecond()-st1);
    
    double st = cpuSecond();
    findTrianglesCPU(&B,&C);
    printf("Time on CPU: %lf sec\n",cpuSecond()-st);

    // for (int i=0;i<9;i++){
    // printf("Sample: %f\n",C.cooValA[i]);
    // }
    
    return 0; 
}
 
 
int findLines(char* fName){
 
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(fName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }
    
    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        i++;
    }
    fclose(fp);
    return i;
}
 
int readCSV(char* fName, cooFormat *A, int* N, int* M, int* nT_Mat, double* matlab_time){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    long* tmp = (long*) malloc(3*sizeof(long));

    /* Constructing the full .csv file names */
    char* csvFileName;
    csvFileName = (char*) malloc(1000*sizeof(char));
    //                                                       B E     C A R E F U L
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Path to file: ~/PD_4/Data/  ! ! ! ! (Change this if data is stored elsewhere)
    strcpy(csvFileName,  "DIMACS10_");
    // Do not change "DataDIMACS10_" unless you want to give it as input name alongside with (auto | great-britain_osm | delaunay_n22)
    // every time
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strcat(csvFileName, fName);    

    char* valFileName;
    valFileName = (char*) malloc(1000*sizeof(char));
    strcat(valFileName, csvFileName);
    strcat(valFileName, "_validation_file.csv");

    strcat(csvFileName, ".csv");

    int leng = findLines(csvFileName);

    printf("Lines in the file are: %d\n",leng);   

    A->nnz = leng;

    /* Allocating memory to hold matrix A = arr */
    A->cooValA = (float*) malloc (leng*sizeof(float));
    A->cooRowIndA = (int*) malloc (leng*sizeof(int));
    A->cooColIndA = (int*) malloc (leng*sizeof(int));

    /* Reading the input data */
    fp = fopen(csvFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    long i = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        // printf("%s", line);
        split(line,",",tmp);
        // printf("I: %d\n",i);
        A->cooRowIndA[i] = tmp[0];
        A->cooColIndA[i] = tmp[1];        
        A->cooValA[i] = tmp[2];
        // printf("A[%d]: %ld\n", j, A[j]);
        
        i++;
    }
/* Close file */
    fclose(fp);

    /* Reading the original matrix N and M, as well as 
    the matlab result and time elapsed */
    fp = fopen(valFileName, "r");
    if (fp == NULL){
        printf("Could not open file\n");
        exit(EXIT_FAILURE);
    }

    if( (read = getline(&line, &len, fp)) != -1 ) {
        char* token = strtok(line, ",");
        char* endPtr;

        (*N) = strtoimax(token,&endPtr,10);
        
        token = strtok(NULL, ",");
        
        (*M) = strtoimax(token,&endPtr,10);

        token = strtok(NULL, ",");

        (*nT_Mat) = strtoimax(token,&endPtr,10);

        token = strtok(NULL, ",");   

        (*matlab_time) = atof(token);
    }

    /* Close file */
    fclose(fp);

    return EXIT_SUCCESS;
}
 
 int split(char* str, char* delim, long* tmp){
     int i = 0;
     char* token = strtok(str, delim);
     char* strNum;
     char* endPtr;
     while (token != NULL) {
         strNum = trim_space(token);
         // printf("Number: %ld\n", strtoimax(strNum,&endPtr,10));
         tmp[i++] = strtoimax(strNum,&endPtr,10);
         // printf("Number: %ld\n", tmp[i-1]);
         token = strtok(NULL, delim);   
     }
     return i;
 }
 
 char *trim_space(char *in){
     char *out = NULL;
     int len;
     if (in) {
         len = strlen(in);
         while(len && isspace(in[len - 1])) --len;
         while(len && *in && isspace(*in)) ++in, --len;
         if (len) {
             out = strndup(in, len);
         }
     }
     return out;
 }
 
 