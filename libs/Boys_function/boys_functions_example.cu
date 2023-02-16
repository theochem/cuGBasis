#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (CUDART_VERSION < 6000)
#error This code requires CUDA 6.0 or later
#else // CUDART_VERSION < 6000 
#include "boys_functions.h"

/* Compute boys(m-1,a) by backward recurrence from boys(m,a) */
__device__ double boys_backward_recurrence (int m, double a)
{
    return fma (2.0*a, boys (m, a), exp (-a)) / (2*m-1);
}

void __global__ boys_functions_kernel (void)
{
    double b0_10 = boys0 (10.0);
    double b0_10g = boys (0, 10.0);
    double b30_26 = boys (30, 26.0);
    double b30_26r = boys_backward_recurrence (31,26.0);

    printf ("\nCompute Boys function of order zero via specific and generic functions\n");
    printf ("----------------------------------------------------------------------\n");
    printf ("boys0(10.0)   = %23.16e\n", b0_10);
    printf ("boys(0,10.0)  = %23.16e\n", b0_10g);

    printf ("\nCompute boys(30,x) directly\n");
    printf ("---------------------------\n");
    printf ("boys(30,26.0) = %23.16e\n", b30_26);

    printf ("\nCompute boys(30,x) via backward recurrence from boys(31,x)\n");
    printf ("----------------------------------------------------------\n");
    printf ("boys(30,26.0) = %23.16e\n", b30_26r);
}

int main (void)
{
    boys_functions_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
#endif // CUDART_VERSION < 6000
