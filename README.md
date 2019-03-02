# PD4
The final assignment

 - Folder /CPU Implementation containts .c code for the CPU solution to the problem

 - Folder /GPU Implementation containts .cu code for the CUDA solution to the problem

 - Folder /Naive GPU Implementation containts .cu code for the naive CUDA solution to the problem ( which ended up to be the best )

 - Folder /cuSPARSE Implementations containts .cu code utilizing the cuSPARSE library for the solution to the problem

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

2) Execute the findTriangles.m script 
	/ Alternatively, you can download the input files straight from the link provided into the report

3) Create a folder called Data into the main folder of the repo and move the input file in there
	/ Alternatively, you can change appropriately line 62: "strcpy(csvFileName,  "../Data/DataDIMACS10_");"" of every readCSV.c/cu file and set the path to the the desired one - not recommended )
( if all the previous are ommited, there will be no input file for the code to use, thus it won't operate ) 

4) Navigate to the folder of any implementation

5) Make

6) Execute ( ex. ./trianglesCPU auto OR ./trianglesGPU delaunay_n10 ) 
