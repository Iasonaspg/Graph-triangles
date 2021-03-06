/********************************************************************************
 *
 * sparse.cu -- Tester function for the csrgemm() CUDA function
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 ********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <sys/time.h>
#include "readCSV.h"
#include "cuFindTriangles.h"
#include "validation.h"


int main(int argc, char** argv){

    int *h_nT, nT_Mat, N, M,
        *d_nT;

    double matlab_time;

    char csrFormatValidationFlag = 0;

    /* Create the struct of type csr Format to hold the Sparse Matrices A and B */
    csrFormat h_A, d_A, d_B;

    /* Validation Only Variables */
    cooFormat h_A_COO_Mat;

    /* Parsing input arguments */
    if ( argc < 4 ) {
        printf("--Reading Input Data from CSV file: Started--\n");    
        readCSV(argv[1], &h_A, &N, &M, &nT_Mat, &matlab_time);
        printf("--Reading Input Data from CSV file: DONE!--\n");   
        if ( argc == 3 )
            if ( strcmp(argv[2], "--csrVal") == 0 )              
            // -------- Do not use when timing --------
                csrFormatValidationFlag = 1; 
    } else {
        printf("Usage: ./trianglesGPU <CSVfileName> <--csrVal>\n");
        printf(" where <CSVfileName>.csv is the name of the input data file (auto | great-britain_osm | delaunay_n22 | delaunay_n10)\n");
        printf("No need for suffix '.csv'\n");
        printf("where <--csrVal> is the verbose flag for validation at quite every stage of the program\n");
        exit(1);
    }


    /* CUDA Device setup */
    size_t threadsPerBlock, warp;
    size_t numberOfBlocks, SMs;
    cudaError_t err;
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    warp = props.warpSize;
    SMs = props.multiProcessorCount;
    // 1D threads & blocks
    threadsPerBlock = 8 * warp;
    numberOfBlocks  = 5 * SMs;


    /* Create the cuSPARSE handle */
    cusparseHandle_t handle = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));    

    /* Construct a descriptor of the matrix A */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    /* Construct a descriptor of the matrix B */
    cusparseMatDescr_t descrB = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate device memory to store the sparse CSR representation of A */
    d_A.nnz = h_A.nnz;
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrVal),    sizeof(float) * d_A.nnz));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrRowPtr), sizeof(int) * (N + 1)));
    CUDA_CALL(cudaMalloc((void **)&(d_A.csrColInd), sizeof(int) * d_A.nnz));

    /* Copy the sparse CSR representation of A from the Host to the Device */
    CUDA_CALL(cudaMemcpy(d_A.csrVal,    h_A.csrVal,    d_A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A.csrRowPtr, h_A.csrRowPtr, (N + 1) * sizeof(int)  , cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_A.csrColInd, h_A.csrColInd, d_A.nnz * sizeof(int)  , cudaMemcpyHostToDevice));
    
    if ( csrFormatValidationFlag ) {

        h_A_COO_Mat.nnz = h_A.nnz;
        // Read from .csv A in COO format as found through Matlab
        printf("--Reading Input Data from CSV file for CSR Format Validation: Started--\n");    
        readCSV_COO( argv[1], &h_A_COO_Mat );        
        printf("--Reading Input Data from CSV file for CSR Format Validation: DONE!--\n");   
        // Validate CSR format of A due to Matlab
        assert( validateCSR( handle, d_A, h_A_COO_Mat, N ) );
    }

    int baseB;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &(d_B.nnz);
    CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    // Allocate device memory to store the Row Pointers of the sparse CSR representation of B
    CUDA_CALL(cudaMalloc((void**)&(d_B.csrRowPtr), sizeof(int)*(N+1)));


    /* Timer variables setup */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

            /* Begin timer */
            cudaEventRecord(start);

    /* First determine the nnz of Sparse Matrix B */
    CHECK_CUSPARSE(cusparseXcsrgemmNnz(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        N, 
                        N, 
                        N,
                        descrA, 
                        d_A.nnz,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrA, 
                        d_A.nnz,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrB, 
                        d_B.csrRowPtr,
                        nnzTotalDevHostPtr ));

    if (NULL != nnzTotalDevHostPtr){
        d_B.nnz = *nnzTotalDevHostPtr;
    } else {
        CUDA_CALL(cudaMemcpy(&(d_B.nnz), d_B.csrRowPtr+N, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&baseB,     d_B.csrRowPtr,   sizeof(int), cudaMemcpyDeviceToHost));
        d_B.nnz -= baseB;
    }


    /* Allocate device memory to store the rest of the sparse CSR representation of B */
    CUDA_CALL(cudaMalloc((void**)&d_B.csrVal,    sizeof(float) * d_B.nnz));
    CUDA_CALL(cudaMalloc((void**)&d_B.csrColInd, sizeof(int) * d_B.nnz));

    /* Perform the actual multiplication A * A = B */
    CHECK_CUSPARSE(cusparseScsrgemm(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        N, 
                        N, 
                        N,
                        descrA, 
                        d_A.nnz,
                        d_A.csrVal,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrA, 
                        d_A.nnz,
                        d_A.csrVal,
                        d_A.csrRowPtr, 
                        d_A.csrColInd,
                        descrB,
                        d_B.csrVal,
                        d_B.csrRowPtr, 
                        d_B.csrColInd ));

    /* Allocating memory to hold the nT variable (number of Triangles) */
    CUDA_CALL(cudaMalloc(&d_nT, 1 * sizeof(int)));
    h_nT = (int*)malloc (1 * sizeof(int));

    /* Zero out the content of the variable, 
    so that the summation result is valid */
    cuZeroVariable<<<1,1>>>( d_nT );

    CUDA_CALL(cudaDeviceSynchronize());

    /* Hadamard Product Manually */
    // Calculating the Number of Triangles (nT) through the kernel 
    // (Only the sumation is performed here, the *(1/6) will be executed later on)
    cuTrianglesFinderHadamardOnly<<<numberOfBlocks, threadsPerBlock>>>
    (d_A, d_B, N, d_nT);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
                       __FILE__,__LINE__);
                return EXIT_FAILURE;
            }

    CUDA_CALL(cudaDeviceSynchronize());

            /* Stop Timer */
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

    /* Copying nT, as calculated on GPU, back to the CPU */
    CUDA_CALL(cudaMemcpy(h_nT, d_nT, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    /* Validating the result */
    // Executing the nT = nT*(1/6), that was omitted in cuFindTriangles
    int pass = validation(*h_nT/3, nT_Mat);    
    // though as of condition if ( col>row ) we have cut additions to half
    // due to the symmetry of the adjacency matrix, so 2*(nT/6) = nT/3
    assert(pass != 0);

    /* Calculate elapsed time */
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* Timer display */
    printf("  -GPU number of triangles nT: %d, Wall clock time: %fms ( < %lf ( Matlab Time ) )\n", *h_nT/3, milliseconds, matlab_time);

            /* Write the results into file */
            FILE *fp;
            fp = fopen("GPU_Results.txt", "a");
            if ( fp == NULL ) {
              perror("Failed: Opening file Failed\n");
              return 1;
            }
            fprintf(fp, "%f\n", milliseconds);
            fclose(fp);

    /* Cleanup */
    CUDA_CALL(cudaFree(d_A.csrVal));          CUDA_CALL(cudaFree(d_B.csrVal));
    CUDA_CALL(cudaFree(d_A.csrRowPtr));       CUDA_CALL(cudaFree(d_B.csrRowPtr));
    CUDA_CALL(cudaFree(d_A.csrColInd));       CUDA_CALL(cudaFree(d_B.csrColInd));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
