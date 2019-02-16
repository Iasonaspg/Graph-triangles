/**********************************************************************
 *
 * readCSV.c -- readCSV function for reading the input from
 *              a .csv file
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef READ_CSV_H
#define READ_CSV_H

int readCSV(char* fName, struct Sparse_Matrix_in_COO_format *A, long* N, long* M, long* nT_Mat, double* matlab_time);

#endif /* READ_CSV_H */