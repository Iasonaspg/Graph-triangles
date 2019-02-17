/**********************************************************************
 *
 * validation.cu (.c) -- Validation function for the number of triangles 
 *                    calculated
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "validation.h"

int validation(long nT, long nT_Mat)
{

  if ( nT != nT_Mat )
  {
    printf("Validation FAILED: nT = %ld, while correct value nT_Mat = %ld \n", nT, nT_Mat);
    return 0;
  }

  return 1;
}