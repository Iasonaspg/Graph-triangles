# Graph-triangles
This project contains our implementations of an algorithm that calculates the number of triangles on an undirected graph. We used both the cuSPARSE library and custom CUDA implementations. The later proved quite faster.

## Contents

 - Folder /1. cuSPARSE Implementations containts .cu code utilizing the cuSPARSE library for the solution to the problem

 - Folder /2. CPU Implementation containts .c code for the CPU solution to the problem

 - Folder /3. GPU Implementations containts .cu code for the CUDA solutions to the problem

 - Folder /Data containts the input files ( has to be created, or the code will have no input )

 - Folder /Data COO Format containts the input files in COO Format ( has to be created, or the code will have no input ) 


## Fixes and Changes on Matlab script ( findTriangles.m )

The Matlab script that was given to us contained minor errors, plus we added some changes to save the data into a .csv file, so that it would be easier to be read in C. Specifically:

1. Changed *http* link to *https*

2. Set *filesep* to `"/"`

3. Saved the input sparse matrix into CSR Format in a .csv file

4. Saved N, M, nT of Matlab and the matlab time into a validation file

## ReadMe - How to execute the code

1) Into findTriangles.m define the folderPath to the folder where the input data files will be created

2) Execute the findTriangles.m script or download the input CSV files straight from the link provided into the report

3) Store inside folder *Data* the necessary CSV files. 

   Alternatively, you can change appropriately line 36: `strcpy(csvFileName,  "../Data/DIMACS10_");` of every readCSV.c/cu file   and set the path to the the desired one - **not recommended** )
   
   **If all the previous are ommited, there will be no input file for the code to use, thus it won't operate**

4) Navigate to the folder of any implementation

5) Make

6) Execute ( ex. `./trianglesCUSPARSE auto --fullVal` OR `./trianglesCPU auto` OR `./trianglesGPU delaunay_n22` ) 
