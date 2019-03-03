/**********************************************************************
 *
 * validation.cu (.c) -- Validation function for the number of triangles 
 *                    calculated
 *
 * Michail Iason Pavlidis <michailpg@ece.auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int validation(int nT, int nT_Mat)
{

  if ( nT != nT_Mat )
  {
    printf("Validation FAILED: nT = %d, while correct value nT_Mat = %d \n", nT, nT_Mat);
    return 0;
  }

  return 1;
}