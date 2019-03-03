/**********************************************************************
 *
 * validation.h -- Validation function for the number of triangles 
 * 					  calculated
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "readCSV.h"

#ifndef VALIDATION_H
#define VALIDATION_H

int validation(int nT, int nT_Mat);

int validateCSR( cusparseHandle_t handle, csrFormat d_A_CSR, cooFormat h_A_COO, int N );

#endif /* VALIDATION_H */
