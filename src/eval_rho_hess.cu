#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "eval_rho_hess.cuh"
#include "cuda_utils.cuh"
#include "cuda_basis_utils.cuh"
#include "basis_to_gpu.cuh"
#include "eval.cuh"


using namespace chemtools;


__device__ __forceinline__ void chemtools::eval_AOs_hess(
          double*  d_AOs_hess,
    const double3  pt,
    const int      n_pts,
    const int      n_cshells,
    uint&          idx,
    const int      iorb_start
) {
    // Setup the initial variables.
    uint ibasis         = 0;                                        // Index to go over constant memory.
    uint iorb           = iorb_start;                               // Index to go over rows of d_contractions_array
    uint n_cshells_todo = (uint) g_constant_basis[ibasis++];        // Number of contracted shells in memory
    
    #pragma unroll 1
    for (int i_shell = 0; i_shell < n_cshells_todo; i_shell++) {
        double3 r_A = {
            pt.x - g_constant_basis[ibasis++],
            pt.y - g_constant_basis[ibasis++],
            pt.z - g_constant_basis[ibasis++]
        };

        uint n_seg_shells = (uint) g_constant_basis[ibasis++];
        uint n_prims      = (uint) g_constant_basis[ibasis++];
        #pragma unroll 1
        for (int iseg = 0; iseg < n_seg_shells; iseg++) {
            int L = (int) g_constant_basis[ibasis + n_prims + (n_prims + 1) * iseg];
            for (int i_prim = 0; i_prim < n_prims; i_prim++) {
              double c = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
              double a  = g_constant_basis[ibasis + i_prim];
              double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
    
              if (L == S_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        2*a*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        4*a*a*r_A.x*r_A.y  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        4*a*a*r_A.x*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        2*a*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        4*a*a*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_s(a) *
                        2*a*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
              }
              if (L == P_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.x*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.y*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.x*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        4*a*a*r_A.x*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.x*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.y*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.x*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        4*a*a*r_A.x*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.y*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        4*a*a*r_A.x*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.x*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_p(a) *
                        2*a*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
              }
              if (L == D_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        2*(2*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 3*a*r_A.x*r_A.x + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        4*a*r_A.x*r_A.y*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        4*a*r_A.x*r_A.z*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        2*a*r_A.x*r_A.x*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        4*a*a*r_A.x*r_A.x*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_d(a, 2, 0, 0) *
                        2*a*r_A.x*r_A.x*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        2*a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        4*a*r_A.x*r_A.y*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        4*a*a*r_A.x*r_A.y*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        2*(2*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 3*a*r_A.y*r_A.y + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        4*a*r_A.y*r_A.z*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_d(a, 0, 2, 0) *
                        2*a*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        2*a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        4*a*a*r_A.x*r_A.y*r_A.z*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        4*a*r_A.x*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        2*a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        4*a*r_A.y*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_d(a, 0, 0, 2) *
                        2*(2*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 3*a*r_A.z*r_A.z + 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        2*a*r_A.x*r_A.y*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        (-2*a*r_A.x*r_A.x + 2*a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        2*a*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        2*a*r_A.x*r_A.y*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        2*a*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_d(a, 1, 1, 0) *
                        2*a*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        2*a*r_A.x*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        2*a*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        (-2*a*r_A.x*r_A.x + 2*a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        2*a*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        2*a*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_d(a, 1, 0, 1) *
                        2*a*r_A.x*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        2*a*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        2*a*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        2*a*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        2*a*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        (-2*a*r_A.y*r_A.y + 2*a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 5) + idx] +=
                    normalization_primitive_d(a, 0, 1, 1) *
                        2*a*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
              }
              if (L == DP_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of (-x**2 - y**2 + 2*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-2*a*r_A.x*r_A.x*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1) + 2*a*r_A.x*r_A.x + a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.x*r_A.y*(-a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) + 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.x*r_A.z*(-a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-2*a*r_A.y*r_A.y*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1) + 2*a*r_A.y*r_A.y + a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.y*r_A.z*(-a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-2*a*r_A.z*r_A.z*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) + 2) - 4*a*r_A.z*r_A.z + a*(r_A.x*r_A.x + r_A.y*r_A.y - 2*r_A.z*r_A.z) + 2)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*x*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.z*(6.92820323027551*a*r_A.x*r_A.x - 10.3923048454133)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.y*r_A.z*(3.46410161513775*a*r_A.x*r_A.x - 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-3.46410161513775*a*r_A.x*r_A.x + 2*a*r_A.z*r_A.z*(3.46410161513775*a*r_A.x*r_A.x - 1.73205080756888) + 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.z*(6.92820323027551*a*r_A.y*r_A.y - 3.46410161513775)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.y*(6.92820323027551*a*r_A.z*r_A.z - 3.46410161513775)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.z*(6.92820323027551*a*r_A.z*r_A.z - 10.3923048454133)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*y*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.y*r_A.z*(6.92820323027551*a*r_A.x*r_A.x - 3.46410161513775)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.z*(6.92820323027551*a*r_A.y*r_A.y - 3.46410161513775)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.y*(6.92820323027551*a*r_A.z*r_A.z - 3.46410161513775)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.y*r_A.z*(6.92820323027551*a*r_A.y*r_A.y - 10.3923048454133)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-3.46410161513775*a*r_A.y*r_A.y + 2*a*r_A.z*r_A.z*(3.46410161513775*a*r_A.y*r_A.y - 1.73205080756888) + 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.y*r_A.z*(6.92820323027551*a*r_A.z*r_A.z - 10.3923048454133)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of (1.73205080756888*x**2 - 1.73205080756888*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (3.46410161513775*a*r_A.x*r_A.x*(a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1) - 3.46410161513775*a*r_A.x*r_A.x - 1.73205080756888*a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        3.46410161513775*a*a*r_A.x*r_A.y*(r_A.x*r_A.x - r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        3.46410161513775*a*r_A.x*r_A.z*(a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (3.46410161513775*a*r_A.y*r_A.y*(a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1) + 3.46410161513775*a*r_A.y*r_A.y - 1.73205080756888*a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        3.46410161513775*a*r_A.y*r_A.z*(a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*(r_A.x*r_A.x - r_A.y*r_A.y)*(3.46410161513775*a*r_A.z*r_A.z - 1.73205080756888)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*x*y*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.y*(6.92820323027551*a*r_A.x*r_A.x - 10.3923048454133)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        (-3.46410161513775*a*r_A.x*r_A.x + 2*a*r_A.y*r_A.y*(3.46410161513775*a*r_A.x*r_A.x - 1.73205080756888) + 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.y*r_A.z*(3.46410161513775*a*r_A.x*r_A.x - 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.y*(6.92820323027551*a*r_A.y*r_A.y - 10.3923048454133)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        2*a*r_A.x*r_A.z*(3.46410161513775*a*r_A.y*r_A.y - 1.73205080756888)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_pure_d(a) *
                        a*r_A.x*r_A.y*(6.92820323027551*a*r_A.z*r_A.z - 3.46410161513775)  *
                        ce;
              }
              if (L == F_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        2*r_A.x*(a*r_A.x*r_A.x*(2*a*r_A.x*r_A.x - 3) - 4*a*r_A.x*r_A.x + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        2*a*r_A.x*r_A.x*r_A.y*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        2*a*r_A.x*r_A.x*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        2*a*r_A.x*r_A.x*r_A.x*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        4*a*a*r_A.x*r_A.x*r_A.x*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_f(a, 3, 0, 0) *
                        2*a*r_A.x*r_A.x*r_A.x*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        2*a*r_A.y*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        2*a*r_A.x*r_A.y*r_A.y*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        4*a*a*r_A.x*r_A.y*r_A.y*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        2*r_A.y*(a*r_A.y*r_A.y*(2*a*r_A.y*r_A.y - 3) - 4*a*r_A.y*r_A.y + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        2*a*r_A.y*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_f(a, 0, 3, 0) *
                        2*a*r_A.y*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        2*a*r_A.z*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        4*a*a*r_A.x*r_A.y*r_A.z*r_A.z*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        2*a*r_A.x*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        2*a*r_A.z*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        2*a*r_A.y*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_f(a, 0, 0, 3) *
                        2*r_A.z*(a*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3) - 4*a*r_A.z*r_A.z + 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        2*a*r_A.x*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        2*r_A.y*(2*a*r_A.x*r_A.x - 1)*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        2*a*r_A.y*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        2*r_A.x*(2*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 3*a*r_A.y*r_A.y + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_f(a, 1, 2, 0) *
                        2*a*r_A.x*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        2*r_A.y*(2*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 3*a*r_A.x*r_A.x + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        2*r_A.x*(a*r_A.x*r_A.x - 1)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        2*a*r_A.x*r_A.x*r_A.y*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        2*a*r_A.x*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_f(a, 2, 1, 0) *
                        2*a*r_A.x*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        2*r_A.z*(2*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 3*a*r_A.x*r_A.x + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        2*r_A.x*(a*r_A.x*r_A.x - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        2*a*r_A.x*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        2*a*r_A.x*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 5) + idx] +=
                    normalization_primitive_f(a, 2, 0, 1) *
                        2*a*r_A.x*r_A.x*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        2*a*r_A.x*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        2*a*r_A.y*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        2*r_A.z*(2*a*r_A.x*r_A.x - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        2*a*r_A.x*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 6) + idx] +=
                    normalization_primitive_f(a, 1, 0, 2) *
                        2*r_A.x*(2*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 3*a*r_A.z*r_A.z + 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        2*a*r_A.y*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        2*a*r_A.x*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        2*a*r_A.y*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        2*r_A.z*(2*a*r_A.y*r_A.y - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 7) + idx] +=
                    normalization_primitive_f(a, 0, 1, 2) *
                        2*r_A.y*(2*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 3*a*r_A.z*r_A.z + 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        2*a*r_A.y*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        4*a*r_A.x*r_A.y*r_A.z*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        2*a*r_A.x*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        2*r_A.z*(2*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 3*a*r_A.y*r_A.y + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        2*r_A.y*(a*r_A.y*r_A.y - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 8) + idx] +=
                    normalization_primitive_f(a, 0, 2, 1) *

                        2*a*r_A.y*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        2*a*r_A.x*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        r_A.z*(2*a*r_A.x*r_A.x - 1)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        r_A.y*(2*a*r_A.x*r_A.x - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        2*a*r_A.x*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        r_A.x*(2*a*r_A.y*r_A.y - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 9) + idx] +=
                    normalization_primitive_f(a, 1, 1, 1) *

                        2*a*r_A.x*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
              }
              if (L == SF_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*(-3*x**2 - 3*y**2 + 2*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(-2*a*r_A.x*r_A.x*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) + 6*a*r_A.x*r_A.x + a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        2*a*r_A.x*r_A.y*r_A.z*(-a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) + 6)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(-2*a*r_A.z*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) - 4*a*r_A.z*r_A.z + a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(-2*a*r_A.y*r_A.y*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) + 6*a*r_A.y*r_A.y + a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(-2*a*r_A.z*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) - 4*a*r_A.z*r_A.z + a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(-6*a*a*r_A.x*r_A.x*r_A.z*r_A.z - 6*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 4*a*a*pow(r_A.z, 4) + 9*a*r_A.x*r_A.x + 9*a*r_A.y*r_A.y - 14*a*r_A.z*r_A.z + 6)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*(-1.22474487139159*x**2 - 1.22474487139159*y**2 + 4.89897948556636*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        1.0*r_A.x*(-2.44948974278318*a*a*pow(r_A.x, 4) - 2.44948974278318*a*a*r_A.x*r_A.x*r_A.y*r_A.y + 9.79795897113271*a*a*r_A.x*r_A.x*r_A.z*r_A.z + 8.57321409974112*a*r_A.x*r_A.x + 3.67423461417477*a*r_A.y*r_A.y - 14.6969384566991*a*r_A.z*r_A.z - 3.67423461417477)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(2.44948974278318*a*r_A.x*r_A.x - 2*a*(a*r_A.x*r_A.x*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.83711730708738*r_A.x*r_A.x - 0.612372435695794*r_A.y*r_A.y + 2.44948974278318*r_A.z*r_A.z) - 1.22474487139159)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(-9.79795897113271*a*r_A.x*r_A.x - 2*a*(a*r_A.x*r_A.x*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.83711730708738*r_A.x*r_A.x - 0.612372435695794*r_A.y*r_A.y + 2.44948974278318*r_A.z*r_A.z) + 4.89897948556636)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(-2*a*r_A.y*r_A.y*(a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159) + 2.44948974278318*a*r_A.y*r_A.y + a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*r_A.y*r_A.z*(-2*a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 7.34846922834953)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(-2*a*r_A.z*r_A.z*(a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) + 4.89897948556636) - 9.79795897113271*a*r_A.z*r_A.z + a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) + 4.89897948556636)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*(-1.22474487139159*x**2 - 1.22474487139159*y**2 + 4.89897948556636*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(-2*a*r_A.x*r_A.x*(a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159) + 2.44948974278318*a*r_A.x*r_A.x + a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(-2*a*r_A.y*r_A.y*(a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159) + 2.44948974278318*a*r_A.y*r_A.y + a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 1.22474487139159)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*r_A.y*r_A.z*(-2*a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 7.34846922834953)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        1.0*r_A.y*(-2.44948974278318*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 2.44948974278318*a*a*pow(r_A.y, 4) + 9.79795897113271*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 3.67423461417477*a*r_A.x*r_A.x + 8.57321409974112*a*r_A.y*r_A.y - 14.6969384566991*a*r_A.z*r_A.z - 3.67423461417477)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(-9.79795897113271*a*r_A.y*r_A.y - 2*a*(a*r_A.y*r_A.y*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) - 0.612372435695794*r_A.x*r_A.x - 1.83711730708738*r_A.y*r_A.y + 2.44948974278318*r_A.z*r_A.z) + 4.89897948556636)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(-2*a*r_A.z*r_A.z*(a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) + 4.89897948556636) - 9.79795897113271*a*r_A.z*r_A.z + a*(1.22474487139159*r_A.x*r_A.x + 1.22474487139159*r_A.y*r_A.y - 4.89897948556636*r_A.z*r_A.z) + 4.89897948556636)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*(3.87298334620742*x**2 - 3.87298334620742*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(7.74596669241483*a*r_A.x*r_A.x*(a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1) - 7.74596669241483*a*r_A.x*r_A.x - 3.87298334620742*a*(r_A.x*r_A.x - r_A.y*r_A.y) + 3.87298334620742)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        7.74596669241483*a*a*r_A.x*r_A.y*r_A.z*(r_A.x*r_A.x - r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(7.74596669241483*a*r_A.z*r_A.z - 3.87298334620742)*(a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(7.74596669241483*a*r_A.y*r_A.y*(a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1) + 7.74596669241483*a*r_A.y*r_A.y - 3.87298334620742*a*(r_A.x*r_A.x - r_A.y*r_A.y) - 3.87298334620742)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(7.74596669241483*a*r_A.z*r_A.z - 3.87298334620742)*(a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.z*(7.74596669241483*a*r_A.z*r_A.z*(r_A.x*r_A.x - r_A.y*r_A.y) - 11.6189500386223*r_A.x*r_A.x + 11.6189500386223*r_A.y*r_A.y)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of 3.87298334620742*x*y*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*r_A.y*r_A.z*(15.4919333848297*a*r_A.x*r_A.x - 23.2379000772445)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.z*(7.74596669241483*a*r_A.x*r_A.x - 3.87298334620742)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(7.74596669241483*a*r_A.x*r_A.x - 3.87298334620742)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*r_A.y*r_A.z*(15.4919333848297*a*r_A.y*r_A.y - 23.2379000772445)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(7.74596669241483*a*r_A.y*r_A.y - 3.87298334620742)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*r_A.y*r_A.z*(15.4919333848297*a*r_A.z*r_A.z - 23.2379000772445)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*(1.58113883008419*x**2 - 4.74341649025257*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        1.0*r_A.x*(3.16227766016838*a*a*pow(r_A.x, 4) - 9.48683298050514*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 11.0679718105893*a*r_A.x*r_A.x + 14.2302494707577*a*r_A.y*r_A.y + 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(9.48683298050514*a*r_A.x*r_A.x + 2*a*(a*r_A.x*r_A.x*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y) - 2.37170824512628*r_A.x*r_A.x + 2.37170824512628*r_A.y*r_A.y) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        2*a*r_A.z*(a*r_A.x*r_A.x*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y) - 2.37170824512628*r_A.x*r_A.x + 2.37170824512628*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(2*a*r_A.y*r_A.y*(a*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y) + 4.74341649025257) + 9.48683298050514*a*r_A.y*r_A.y - a*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        2*a*r_A.x*r_A.y*r_A.z*(a*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y) + 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 5) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.x*(1.58113883008419*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*(4.74341649025257*x**2 - 1.58113883008419*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.y*(2*a*r_A.x*r_A.x*(a*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) - 4.74341649025257) - 9.48683298050514*a*r_A.x*r_A.x - a*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) + 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        r_A.x*(2*a*r_A.y*r_A.y*(a*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) - 4.74341649025257) + 3.16227766016838*a*r_A.y*r_A.y - a*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) + 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        2*a*r_A.x*r_A.y*r_A.z*(a*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        1.0*r_A.y*(9.48683298050514*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 3.16227766016838*a*a*pow(r_A.y, 4) - 14.2302494707577*a*r_A.x*r_A.x + 11.0679718105893*a*r_A.y*r_A.y - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        2*a*r_A.z*(a*r_A.y*r_A.y*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y) - 2.37170824512628*r_A.x*r_A.x + 2.37170824512628*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 6) + idx] +=
                    normalization_primitive_pure_f(a) *

                        a*r_A.y*(4.74341649025257*r_A.x*r_A.x - 1.58113883008419*r_A.y*r_A.y)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
              }
              if (L == G_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**4*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        2*a*pow(r_A.z, 4)*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*a*r_A.x*r_A.y*pow(r_A.z, 4)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*r_A.x*r_A.z*r_A.z*r_A.z*(a*r_A.z*r_A.z - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        2*a*pow(r_A.z, 4)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*r_A.y*r_A.z*r_A.z*r_A.z*(a*r_A.z*r_A.z - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        r_A.z*r_A.z*(4*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 2) - 10*a*r_A.z*r_A.z + 12)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        2*a*r_A.y*r_A.z*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        2*a*r_A.x*r_A.z*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        2*a*r_A.x*r_A.y*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        2*a*r_A.y*r_A.z*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        r_A.z*r_A.z*(-6*a*r_A.y*r_A.y + 2*a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_g(a, 0, 1, 3) *

                        2*r_A.y*r_A.z*(a*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3) - 4*a*r_A.z*r_A.z + 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        2*a*r_A.y*r_A.y*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        4*a*r_A.x*r_A.y*r_A.z*r_A.z*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        4*a*r_A.x*r_A.y*r_A.y*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        r_A.z*r_A.z*(4*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 6*a*r_A.y*r_A.y + 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        4*r_A.y*r_A.z*(a*r_A.y*r_A.y - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_g(a, 0, 2, 2) *

                        r_A.y*r_A.y*(4*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 6*a*r_A.z*r_A.z + 2)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**3*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        2*a*r_A.y*r_A.y*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        2*r_A.y*r_A.z*(a*r_A.y*r_A.y*(2*a*r_A.y*r_A.y - 3) - 4*a*r_A.y*r_A.y + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        r_A.y*r_A.y*(-2*a*r_A.y*r_A.y + 2*a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 3) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_g(a, 0, 3, 1) *

                        2*a*r_A.y*r_A.y*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**4*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        2*a*pow(r_A.y, 4)*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        4*a*r_A.x*r_A.y*r_A.y*r_A.y*(a*r_A.y*r_A.y - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        4*a*a*r_A.x*pow(r_A.y, 4)*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        r_A.y*r_A.y*(4*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 2) - 10*a*r_A.y*r_A.y + 12)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        4*a*r_A.y*r_A.y*r_A.y*r_A.z*(a*r_A.y*r_A.y - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_g(a, 0, 4, 0) *

                        2*a*pow(r_A.y, 4)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        2*a*r_A.x*r_A.z*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        2*a*r_A.y*r_A.z*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        r_A.z*r_A.z*(-6*a*r_A.x*r_A.x + 2*a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        2*a*r_A.x*r_A.z*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        2*a*r_A.x*r_A.y*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 5) + idx] +=
                    normalization_primitive_g(a, 1, 0, 3) *

                        2*r_A.x*r_A.z*(a*r_A.z*r_A.z*(2*a*r_A.z*r_A.z - 3) - 4*a*r_A.z*r_A.z + 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        2*a*r_A.x*r_A.y*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        r_A.z*r_A.z*(-2*a*r_A.x*r_A.x + 2*a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        2*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        2*a*r_A.x*r_A.y*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        2*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 6) + idx] +=
                    normalization_primitive_g(a, 1, 1, 2) *

                        2*r_A.x*r_A.y*(2*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 3*a*r_A.z*r_A.z + 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**2*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        2*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        r_A.y*r_A.y*(-2*a*r_A.x*r_A.x + 2*a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        2*r_A.x*r_A.z*(2*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 3*a*r_A.y*r_A.y + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        2*r_A.x*r_A.y*(a*r_A.y*r_A.y - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 7) + idx] +=
                    normalization_primitive_g(a, 1, 2, 1) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**3*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        r_A.y*r_A.y*(-6*a*r_A.x*r_A.x + 2*a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 1) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        2*a*r_A.y*r_A.y*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        2*r_A.x*r_A.y*(a*r_A.y*r_A.y*(2*a*r_A.y*r_A.y - 3) - 4*a*r_A.y*r_A.y + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 8) + idx] +=
                    normalization_primitive_g(a, 1, 3, 0) *

                        2*a*r_A.x*r_A.y*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*z**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        r_A.z*r_A.z*(4*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 6*a*r_A.x*r_A.x + 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        4*a*r_A.x*r_A.y*r_A.z*r_A.z*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        4*r_A.x*r_A.z*(a*r_A.x*r_A.x - 1)*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        2*a*r_A.x*r_A.x*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        4*a*r_A.x*r_A.x*r_A.y*r_A.z*(a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 9) + idx] +=
                    normalization_primitive_g(a, 2, 0, 2) *

                        r_A.x*r_A.x*(4*a*r_A.z*r_A.z*(a*r_A.z*r_A.z - 1) - 6*a*r_A.z*r_A.z + 2)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        2*r_A.y*r_A.z*(2*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 3*a*r_A.x*r_A.x + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        2*r_A.x*r_A.z*(a*r_A.x*r_A.x - 1)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        2*r_A.x*r_A.y*(a*r_A.x*r_A.x - 1)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        2*a*r_A.x*r_A.x*r_A.y*r_A.z*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        r_A.x*r_A.x*(-2*a*r_A.y*r_A.y + 2*a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y - 1) + 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 10) + idx] +=
                    normalization_primitive_g(a, 2, 1, 1) *

                        2*a*r_A.x*r_A.x*r_A.y*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y**2*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        r_A.y*r_A.y*(4*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 1) - 6*a*r_A.x*r_A.x + 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        4*r_A.x*r_A.y*(a*r_A.x*r_A.x - 1)*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        4*a*r_A.x*r_A.y*r_A.y*r_A.z*(a*r_A.x*r_A.x - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        r_A.x*r_A.x*(4*a*r_A.y*r_A.y*(a*r_A.y*r_A.y - 1) - 6*a*r_A.y*r_A.y + 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        4*a*r_A.x*r_A.x*r_A.y*r_A.z*(a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 11) + idx] +=
                    normalization_primitive_g(a, 2, 2, 0) *

                        2*a*r_A.x*r_A.x*r_A.y*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*z*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        2*r_A.x*r_A.z*(a*r_A.x*r_A.x*(2*a*r_A.x*r_A.x - 3) - 4*a*r_A.x*r_A.x + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        2*a*r_A.x*r_A.x*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        r_A.x*r_A.x*(-2*a*r_A.x*r_A.x + 2*a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x - 3) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 12) + idx] +=
                    normalization_primitive_g(a, 3, 0, 1) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.z*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*y*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        2*r_A.x*r_A.y*(a*r_A.x*r_A.x*(2*a*r_A.x*r_A.x - 3) - 4*a*r_A.x*r_A.x + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        r_A.x*r_A.x*(-2*a*r_A.x*r_A.x + 2*a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x - 3) + 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        2*a*r_A.x*r_A.x*r_A.y*r_A.z*(2*a*r_A.x*r_A.x - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.y*(2*a*r_A.y*r_A.y - 3)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.z*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 13) + idx] +=
                    normalization_primitive_g(a, 3, 1, 0) *

                        2*a*r_A.x*r_A.x*r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**4*exp(-a*(x**2 + y**2 + z**2))
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        r_A.x*r_A.x*(4*a*r_A.x*r_A.x*(a*r_A.x*r_A.x - 2) - 10*a*r_A.x*r_A.x + 12)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*r_A.x*r_A.x*r_A.x*r_A.y*(a*r_A.x*r_A.x - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*r_A.x*r_A.x*r_A.x*r_A.z*(a*r_A.x*r_A.x - 2)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        2*a*pow(r_A.x, 4)*(2*a*r_A.y*r_A.y - 1)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        4*a*a*pow(r_A.x, 4)*r_A.y*r_A.z  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 14) + idx] +=
                    normalization_primitive_g(a, 0, 0, 4) *

                        2*a*pow(r_A.x, 4)*(2*a*r_A.z*r_A.z - 1)  *
                        ce;
              }
              if (L == SG_TYPE) {
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of (35*z**4 - 30*z**2*(x**2 + y**2 + z**2) + 3*(x**2 + y**2 + z**2)**2)*exp(-a*(x**2 + y**2 + z**2))/8
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (2*a*r_A.x*r_A.x*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 6*r_A.x*r_A.x - 6*r_A.y*r_A.y + 24*r_A.z*r_A.z) - a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) + 12*r_A.x*r_A.x*(a*(-r_A.x*r_A.x - r_A.y*r_A.y + 4*r_A.z*r_A.z) + 1) + 6*r_A.x*r_A.x + 6*r_A.y*r_A.y - 24*r_A.z*r_A.z)/4  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(6*a*(-r_A.x*r_A.x - r_A.y*r_A.y + 4*r_A.z*r_A.z) + a*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 6*r_A.x*r_A.x - 6*r_A.y*r_A.y + 24*r_A.z*r_A.z) + 6)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(-8*a*(-3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 2*r_A.z*r_A.z) + a*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 6*r_A.x*r_A.x - 6*r_A.y*r_A.y + 24*r_A.z*r_A.z) - 24)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (2*a*r_A.y*r_A.y*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 6*r_A.x*r_A.x - 6*r_A.y*r_A.y + 24*r_A.z*r_A.z) - a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) + 6*r_A.x*r_A.x + 12*r_A.y*r_A.y*(a*(-r_A.x*r_A.x - r_A.y*r_A.y + 4*r_A.z*r_A.z) + 1) + 6*r_A.y*r_A.y - 24*r_A.z*r_A.z)/4  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(-8*a*(-3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 2*r_A.z*r_A.z) + a*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 6*r_A.x*r_A.x - 6*r_A.y*r_A.y + 24*r_A.z*r_A.z) - 24)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 0) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (2*a*r_A.z*r_A.z*(a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) + 24*r_A.x*r_A.x + 24*r_A.y*r_A.y - 16*r_A.z*r_A.z) - a*(35*pow(r_A.z, 4) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) + 3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) - 24*r_A.x*r_A.x - 24*r_A.y*r_A.y - 16*r_A.z*r_A.z*(a*(-3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 2*r_A.z*r_A.z) - 2) + 16*r_A.z*r_A.z)/4  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*(-4.74341649025257*x**2 - 4.74341649025257*y**2 + 6.32455532033676*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.x*r_A.z*(-18.9736659610103*a*a*pow(r_A.x, 4) - 18.9736659610103*a*a*r_A.x*r_A.x*r_A.y*r_A.y + 25.298221281347*a*a*r_A.x*r_A.x*r_A.z*r_A.z + 66.407830863536*a*r_A.x*r_A.x + 28.4604989415154*a*r_A.y*r_A.y - 37.9473319220206*a*r_A.z*r_A.z - 28.4604989415154)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(18.9736659610103*a*r_A.x*r_A.x - 2*a*(2*a*r_A.x*r_A.x*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 14.2302494707577*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y + 6.32455532033676*r_A.z*r_A.z) - 9.48683298050514)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (a*r_A.x*r_A.x*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 14.2302494707577*r_A.x*r_A.x - 4.74341649025257*r_A.y*r_A.y + 6.32455532033676*r_A.z*r_A.z) - 7.11512473537885*r_A.x*r_A.x - 2.37170824512628*r_A.y*r_A.y + r_A.z*r_A.z*(-25.298221281347*a*r_A.x*r_A.x + 12.6491106406735)/2 + 3.16227766016838*r_A.z*r_A.z)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(-2*a*r_A.y*r_A.y*(a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257) + 9.48683298050514*a*r_A.y*r_A.y + a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(-2*a*r_A.z*r_A.z*(a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257) - 12.6491106406735*a*r_A.z*r_A.z + a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 1) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.x*r_A.z*(-18.9736659610103*a*a*r_A.x*r_A.x*r_A.z*r_A.z - 18.9736659610103*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 25.298221281347*a*a*pow(r_A.z, 4) + 28.4604989415154*a*r_A.x*r_A.x + 28.4604989415154*a*r_A.y*r_A.y - 88.5437744847146*a*r_A.z*r_A.z + 37.9473319220206)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*(-4.74341649025257*x**2 - 4.74341649025257*y**2 + 6.32455532033676*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(-2*a*r_A.x*r_A.x*(a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257) + 9.48683298050514*a*r_A.x*r_A.x + a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(-2*a*r_A.y*r_A.y*(a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257) + 9.48683298050514*a*r_A.y*r_A.y + a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(-2*a*r_A.z*r_A.z*(a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257) - 12.6491106406735*a*r_A.z*r_A.z + a*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.y*r_A.z*(-18.9736659610103*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 18.9736659610103*a*a*pow(r_A.y, 4) + 25.298221281347*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 28.4604989415154*a*r_A.x*r_A.x + 66.407830863536*a*r_A.y*r_A.y - 37.9473319220206*a*r_A.z*r_A.z - 28.4604989415154)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (a*r_A.y*r_A.y*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y*(4.74341649025257*r_A.x*r_A.x + 4.74341649025257*r_A.y*r_A.y - 6.32455532033676*r_A.z*r_A.z) - 4.74341649025257*r_A.x*r_A.x - 14.2302494707577*r_A.y*r_A.y + 6.32455532033676*r_A.z*r_A.z) - 2.37170824512628*r_A.x*r_A.x - 7.11512473537885*r_A.y*r_A.y + r_A.z*r_A.z*(-25.298221281347*a*r_A.y*r_A.y + 12.6491106406735)/2 + 3.16227766016838*r_A.z*r_A.z)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 2) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.y*r_A.z*(-18.9736659610103*a*a*r_A.x*r_A.x*r_A.z*r_A.z - 18.9736659610103*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 25.298221281347*a*a*pow(r_A.z, 4) + 28.4604989415154*a*r_A.x*r_A.x + 28.4604989415154*a*r_A.y*r_A.y - 88.5437744847146*a*r_A.z*r_A.z + 37.9473319220206)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of (x**2 - y**2)*(-2.23606797749979*x**2 - 2.23606797749979*y**2 + 13.4164078649987*z**2)*exp(-a*(x**2 + y**2 + z**2))/4
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (-2.23606797749979*a*a*pow(r_A.x, 6) + 13.4164078649987*a*a*pow(r_A.x, 4)*r_A.z*r_A.z + 2.23606797749979*a*a*r_A.x*r_A.x*pow(r_A.y, 4) - 13.4164078649987*a*a*r_A.x*r_A.x*r_A.y*r_A.y*r_A.z*r_A.z + 10.0623058987491*a*pow(r_A.x, 4) - 33.5410196624969*a*r_A.x*r_A.x*r_A.z*r_A.z - 1.11803398874989*a*pow(r_A.y, 4) + 6.70820393249937*a*r_A.y*r_A.y*r_A.z*r_A.z - 6.70820393249937*r_A.x*r_A.x + 6.70820393249937*r_A.z*r_A.z)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        a*r_A.x*r_A.y*(-2*a*(r_A.x*r_A.x - r_A.y*r_A.y)*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 8.94427190999916*r_A.x*r_A.x - 8.94427190999916*r_A.y*r_A.y)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(-26.8328157299975*a*(r_A.x*r_A.x - r_A.y*r_A.y) - 2*a*(a*(r_A.x*r_A.x - r_A.y*r_A.y)*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - 4.47213595499958*r_A.x*r_A.x + 13.4164078649987*r_A.z*r_A.z) + 26.8328157299975)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (-2.23606797749979*a*a*pow(r_A.x, 4)*r_A.y*r_A.y + 13.4164078649987*a*a*r_A.x*r_A.x*r_A.y*r_A.y*r_A.z*r_A.z + 2.23606797749979*a*a*pow(r_A.y, 6) - 13.4164078649987*a*a*pow(r_A.y, 4)*r_A.z*r_A.z + 1.11803398874989*a*pow(r_A.x, 4) - 6.70820393249937*a*r_A.x*r_A.x*r_A.z*r_A.z - 10.0623058987491*a*pow(r_A.y, 4) + 33.5410196624969*a*r_A.y*r_A.y*r_A.z*r_A.z + 6.70820393249937*r_A.y*r_A.y - 6.70820393249937*r_A.z*r_A.z)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(-26.8328157299975*a*(r_A.x*r_A.x - r_A.y*r_A.y) - 2*a*(a*(r_A.x*r_A.x - r_A.y*r_A.y)*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 4.47213595499958*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - 26.8328157299975)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 3) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (r_A.x*r_A.x - r_A.y*r_A.y)*(-2*a*r_A.z*r_A.z*(a*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 13.4164078649987) - 26.8328157299975*a*r_A.z*r_A.z + a*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 13.4164078649987)/2  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*(-2.23606797749979*x**2 - 2.23606797749979*y**2 + 13.4164078649987*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.x*r_A.y*(-8.94427190999916*a*a*pow(r_A.x, 4) - 8.94427190999916*a*a*r_A.x*r_A.x*r_A.y*r_A.y + 53.665631459995*a*a*r_A.x*r_A.x*r_A.z*r_A.z + 31.3049516849971*a*r_A.x*r_A.x + 13.4164078649987*a*r_A.y*r_A.y - 80.4984471899924*a*r_A.z*r_A.z - 13.4164078649987)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (a*r_A.x*r_A.x*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - a*r_A.y*r_A.y*(2*a*r_A.x*r_A.x*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - 6.70820393249937*r_A.x*r_A.x - 2.23606797749979*r_A.y*r_A.y + 13.4164078649987*r_A.z*r_A.z) - 3.35410196624968*r_A.x*r_A.x + r_A.y*r_A.y*(8.94427190999916*a*r_A.x*r_A.x - 4.47213595499958)/2 - 1.11803398874989*r_A.y*r_A.y + 6.70820393249937*r_A.z*r_A.z)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(-53.665631459995*a*r_A.x*r_A.x - 2*a*(2*a*r_A.x*r_A.x*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - 6.70820393249937*r_A.x*r_A.x - 2.23606797749979*r_A.y*r_A.y + 13.4164078649987*r_A.z*r_A.z) + 26.8328157299975)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.x*r_A.y*(-8.94427190999916*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 8.94427190999916*a*a*pow(r_A.y, 4) + 53.665631459995*a*a*r_A.y*r_A.y*r_A.z*r_A.z + 13.4164078649987*a*r_A.x*r_A.x + 31.3049516849971*a*r_A.y*r_A.y - 80.4984471899924*a*r_A.z*r_A.z - 13.4164078649987)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(-53.665631459995*a*r_A.y*r_A.y - 2*a*(2*a*r_A.y*r_A.y*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) - 2.23606797749979*r_A.x*r_A.x - 6.70820393249937*r_A.y*r_A.y + 13.4164078649987*r_A.z*r_A.z) + 26.8328157299975)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 4) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(-2*a*r_A.z*r_A.z*(a*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 13.4164078649987) - 26.8328157299975*a*r_A.z*r_A.z + a*(2.23606797749979*r_A.x*r_A.x + 2.23606797749979*r_A.y*r_A.y - 13.4164078649987*r_A.z*r_A.z) + 13.4164078649987)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*(4.18330013267038*x**2 - 12.5499003980111*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.x*r_A.z*(16.7332005306815*a*a*pow(r_A.x, 4) - 50.1996015920445*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 58.5662018573853*a*r_A.x*r_A.x + 75.2994023880668*a*r_A.y*r_A.y + 25.0998007960223)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(50.1996015920445*a*r_A.x*r_A.x + 2*a*(2*a*r_A.x*r_A.x*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) - 12.5499003980111*r_A.x*r_A.x + 12.5499003980111*r_A.y*r_A.y) - 25.0998007960223)/2  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (-a*r_A.x*r_A.x*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) + a*r_A.z*r_A.z*(2*a*r_A.x*r_A.x*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) - 12.5499003980111*r_A.x*r_A.x + 12.5499003980111*r_A.y*r_A.y) + 6.27495019900557*r_A.x*r_A.x - 6.27495019900557*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(2*a*r_A.y*r_A.y*(a*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) + 12.5499003980111) + 25.0998007960223*a*r_A.y*r_A.y - a*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) - 12.5499003980111)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)*(a*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y) + 12.5499003980111)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 5) + idx] +=
                    normalization_primitive_pure_g(a) *

                        a*r_A.x*r_A.z*(4.18330013267038*r_A.x*r_A.x - 12.5499003980111*r_A.y*r_A.y)*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*(12.5499003980111*x**2 - 4.18330013267038*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.y*r_A.z*(2*a*r_A.x*r_A.x*(a*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) - 12.5499003980111) - 25.0998007960223*a*r_A.x*r_A.x - a*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) + 12.5499003980111)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.z*(2*a*r_A.y*r_A.y*(a*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) - 12.5499003980111) + 8.36660026534076*a*r_A.y*r_A.y - a*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) + 12.5499003980111)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(2*a*r_A.z*r_A.z - 1)*(a*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) - 12.5499003980111)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        0.5*r_A.y*r_A.z*(50.1996015920445*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 16.7332005306815*a*a*pow(r_A.y, 4) - 75.2994023880668*a*r_A.x*r_A.x + 58.5662018573853*a*r_A.y*r_A.y - 25.0998007960223)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (-a*r_A.y*r_A.y*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) + a*r_A.z*r_A.z*(2*a*r_A.y*r_A.y*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y) - 12.5499003980111*r_A.x*r_A.x + 12.5499003980111*r_A.y*r_A.y) + 6.27495019900557*r_A.x*r_A.x - 6.27495019900557*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 6) + idx] +=
                    normalization_primitive_pure_g(a) *

                        a*r_A.y*r_A.z*(12.5499003980111*r_A.x*r_A.x - 4.18330013267038*r_A.y*r_A.y)*(2*a*r_A.z*r_A.z - 3)  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of (5.91607978309962*x**4 - 35.4964786985977*x**2*y**2 + 5.91607978309962*y**4)*exp(-a*(x**2 + y**2 + z**2))/8
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (2.95803989154981*a*a*pow(r_A.x, 6) - 17.7482393492988*a*a*pow(r_A.x, 4)*r_A.y*r_A.y + 2.95803989154981*a*a*r_A.x*r_A.x*pow(r_A.y, 4) - 13.3111795119741*a*pow(r_A.x, 4) + 44.3705983732471*a*r_A.x*r_A.x*r_A.y*r_A.y - 1.4790199457749*a*pow(r_A.y, 4) + 8.87411967464942*r_A.x*r_A.x - 8.87411967464942*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        r_A.x*r_A.y*(2*a*(8.87411967464942*r_A.x*r_A.x - 2.95803989154981*r_A.y*r_A.y) + 2*a*(2*a*(0.739509972887452*pow(r_A.x, 4) - 4.43705983732471*r_A.x*r_A.x*r_A.y*r_A.y + 0.739509972887452*pow(r_A.y, 4)) - 2.95803989154981*r_A.x*r_A.x + 8.87411967464942*r_A.y*r_A.y) - 17.7482393492988)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        2*a*r_A.x*r_A.z*(2*a*(0.739509972887452*pow(r_A.x, 4) - 4.43705983732471*r_A.x*r_A.x*r_A.y*r_A.y + 0.739509972887452*pow(r_A.y, 4)) - 2.95803989154981*r_A.x*r_A.x + 8.87411967464942*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (2.95803989154981*a*a*pow(r_A.x, 4)*r_A.y*r_A.y - 17.7482393492988*a*a*r_A.x*r_A.x*pow(r_A.y, 4) + 2.95803989154981*a*a*pow(r_A.y, 6) - 1.4790199457749*a*pow(r_A.x, 4) + 44.3705983732471*a*r_A.x*r_A.x*r_A.y*r_A.y - 13.3111795119741*a*pow(r_A.y, 4) - 8.87411967464942*r_A.x*r_A.x + 8.87411967464942*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        2*a*r_A.y*r_A.z*(2*a*(0.739509972887452*pow(r_A.x, 4) - 4.43705983732471*r_A.x*r_A.x*r_A.y*r_A.y + 0.739509972887452*pow(r_A.y, 4)) + 8.87411967464942*r_A.x*r_A.x - 2.95803989154981*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 7) + idx] +=
                    normalization_primitive_pure_g(a) *

                        a*(a*r_A.z*r_A.z*(2.95803989154981*pow(r_A.x, 4) - 17.7482393492988*r_A.x*r_A.x*r_A.y*r_A.y + 2.95803989154981*pow(r_A.y, 4)) - 1.4790199457749*pow(r_A.x, 4) + 8.87411967464942*r_A.x*r_A.x*r_A.y*r_A.y - 1.4790199457749*pow(r_A.y, 4))  *
                        ce;
                // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*(5.91607978309962*x**2 - 5.91607978309962*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
                d_AOs_hess[n_pts * (n_cshells*0 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        1.0*r_A.x*r_A.y*(11.8321595661992*a*a*pow(r_A.x, 4) - 11.8321595661992*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 41.4125584816973*a*r_A.x*r_A.x + 17.7482393492988*a*r_A.y*r_A.y + 17.7482393492988)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*1 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        (11.8321595661992*a*a*pow(r_A.x, 4)*r_A.y*r_A.y - 11.8321595661992*a*a*r_A.x*r_A.x*pow(r_A.y, 4) - 5.91607978309962*a*pow(r_A.x, 4) - 8.88178419700125e-16*a*r_A.x*r_A.x*r_A.y*r_A.y + 5.91607978309962*a*pow(r_A.y, 4) + 8.87411967464942*r_A.x*r_A.x - 8.87411967464942*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*2 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        2*a*r_A.y*r_A.z*(5.91607978309962*a*r_A.x*r_A.x*(r_A.x*r_A.x - r_A.y*r_A.y) - 8.87411967464942*r_A.x*r_A.x + 2.95803989154981*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*3 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        1.0*r_A.x*r_A.y*(11.8321595661992*a*a*r_A.x*r_A.x*r_A.y*r_A.y - 11.8321595661992*a*a*pow(r_A.y, 4) - 17.7482393492988*a*r_A.x*r_A.x + 41.4125584816973*a*r_A.y*r_A.y - 17.7482393492988)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*4 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        2*a*r_A.x*r_A.z*(5.91607978309962*a*r_A.y*r_A.y*(r_A.x*r_A.x - r_A.y*r_A.y) - 2.95803989154981*r_A.x*r_A.x + 8.87411967464942*r_A.y*r_A.y)  *
                        ce;
                  d_AOs_hess[n_pts * (n_cshells*5 + iorb + 8) + idx] +=
                    normalization_primitive_pure_g(a) *

                        a*r_A.x*r_A.y*(r_A.x*r_A.x - r_A.y*r_A.y)*(11.8321595661992*a*r_A.z*r_A.z - 5.91607978309962)  *
                        ce;
              }
    
            } // End going over contractions of a single segmented shell.
            // Update index that goes over each contraction.
            if(L == S_TYPE){
                iorb += 1;
            }
            else if (L == P_TYPE) {
                iorb += 3;
            }
            else if (L == D_TYPE) {
                iorb += 6;
            }
            else if (L == DP_TYPE) {
                iorb += 5;
            }
            else if (L == F_TYPE) {
                iorb += 10;
            }
            else if (L == SF_TYPE) {
                iorb += 7;
            }
            else if (L == G_TYPE) {
                iorb += 15;
            }
            else if (L == SG_TYPE) {
                iorb += 9;
            }
      } // End updating segmented shell.

      // Update index of constant memory, add the number of exponents then number of angular momentum terms then
      //        add the number of coefficients.
      ibasis += n_prims + n_seg_shells + n_seg_shells * n_prims;
    } // End Contractions
}



__global__ void chemtools::eval_AOs_hess_on_any_grid(
          double* __restrict__ d_AO_hess,
    const double* __restrict__ d_points,
    const int     n_pts,
    const int     n_cshells,
    const int     iorb_start
    ){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pts) {
        // Get the grid points where `d_points` is in column-major order with shape (N, 3)
        double pt_x = d_points[idx];
        double pt_y = d_points[idx + n_pts];
        double pt_z = d_points[idx + n_pts * 2];
        // Evaluate the contraction derivatives and store it in d_AO_vals
        eval_AOs_hess(d_AO_hess, {pt_x, pt_y, pt_z}, n_pts, n_cshells, idx, iorb_start);
    }
}


__host__ std::vector<double> chemtools::evaluate_contraction_second_derivative(
    IOData& iodata, const double* h_points, const int n_pts
) {
  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  int knbasisfuncs = molecular_basis.numb_basis_functions();

  // AOs second derivs in column-major order with shape (6, M, N).
  std::vector<double> h_AOs_hess(6 * knbasisfuncs * n_pts);
  
  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  CUDA_CHECK(cudaMalloc((double **) &d_points, sizeof(double) * 3 * n_pts));
  CUDA_CHECK(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * n_pts, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_sec_deriv_contr;
  CUDA_CHECK(cudaMalloc((double **) &d_sec_deriv_contr, sizeof(double) * 6 * n_pts * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((n_pts + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  evaluate_scalar_quantity_density(
      molecular_basis,
      false,
      false,
      "rho_hess",
      d_sec_deriv_contr,
      d_points,
      n_pts,
      knbasisfuncs,
      threadsPerBlock,
      grid
  );

  // Transfer from device memory to host memory
  CUDA_CHECK(cudaMemcpy(&h_AOs_hess[0],
                                          d_sec_deriv_contr,
                                          sizeof(double) * 6 * n_pts * knbasisfuncs, cudaMemcpyDeviceToHost));

  cudaFree(d_points);
  cudaFree(d_sec_deriv_contr);
  return h_AOs_hess;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_hessian(
    IOData& iodata, const double* h_points, int n_pts, bool return_row
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> hessian = evaluate_electron_density_hessian_handle(
      handle, iodata, h_points, n_pts, return_row
  );
  cublasDestroy(handle);
  return hessian;
}


__host__ static __forceinline__ void chemtools::evaluate_first_term(
          cublasHandle_t&            handle,
    const MolecularBasis& basis,
          double*                    d_hessian,
    const double* const              d_pts,
    const double* const              d_one_rdm,
    const double* const              d_all_ones,
          std::vector<double*>&      d_pointers,
    const size_t&                    npts_iter,
    const size_t&                    nbasis)
    {
    // Take the pre-allocated memory
    double* d_AOs_deriv            = d_pointers[0];
    double* d_MOs_derivi           = d_pointers[1];
    double* d_temp_rdm_derivs_copy = d_pointers[2];
    
    // Evaluate AO derivatives, row-order (3, M, N),
    dim3 threadsPerBlock(128);
    dim3 grid((npts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    evaluate_scalar_quantity_density(
        basis,
        false,
        false,
        "rho_deriv",
        d_AOs_deriv,
        d_pts,
        npts_iter,
        nbasis,
        threadsPerBlock,
        grid
    );
    
    // For each derivative, calculate the derivative of electron density seperately.
    int i_sec_derivs = 0;
    #pragma unroll
    for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
        // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
        double *d_ith_deriv = &d_AOs_deriv[i_deriv * npts_iter * nbasis];
        
        // Matrix multiple one-rdm with the ith derivative of contractions
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 npts_iter, nbasis, nbasis,
                                 &alpha, d_ith_deriv, npts_iter,
                                 d_one_rdm, nbasis, &beta,
                                 d_MOs_derivi, npts_iter));
        
        #pragma unroll
        for(int j_deriv = i_deriv; j_deriv < 3; j_deriv++) {
            // Get the jth derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
            double *d_jth_deriv = &d_AOs_deriv[j_deriv * npts_iter * nbasis];

            // Do a hadamard product with the original contractions.
            dim3 threadsPerBlock2(320);
            dim3 grid2((npts_iter * nbasis + threadsPerBlock.x - 1) / (threadsPerBlock.x));
            hadamard_product_outplace<<<grid2, threadsPerBlock2>>>(
                d_temp_rdm_derivs_copy, d_jth_deriv, d_MOs_derivi, nbasis * npts_iter
            );
            
            // Take the sum. This is done via matrix-vector multiplication of ones
            CUBLAS_CHECK(cublasDgemv(
              handle, CUBLAS_OP_N, npts_iter, nbasis,
              &alpha,
              d_temp_rdm_derivs_copy, npts_iter, d_all_ones, 1,
              &beta,
              &d_hessian[i_sec_derivs * npts_iter], 1)
            );
          
            i_sec_derivs += 1;
        }
    }
}


__host__ std::vector<double> chemtools::evaluate_electron_density_hessian_handle(
          cublasHandle_t& handle,
          IOData&         iodata,
    const double*         h_points,
    const int             n_pts,
    const bool            return_row
) {
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    //  Solve for N (12MN + 9N + N + M) * 8 bytes = Free memory (in bytes) 
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return (
              (mem  / sizeof(double))  - numb_basis - numb_basis * numb_basis
              ) /  (8 * numb_basis + 9);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // The Hessian in col-major with shape (N, 3, 3).
    std::vector<double> h_rho_hess(9 * n_pts);
    
    // Allocate some memory outside the loop to avoid the slow cudaMalloc cudaFree
    double *d_one_rdm = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_one_rdm, n_basis * n_basis * sizeof(double)));
    CUBLAS_CHECK(cublasSetMatrix(
        iodata.GetOneRdmShape(),
        iodata.GetOneRdmShape(),
        sizeof(double),
        iodata.GetMOOneRDM(),
        iodata.GetOneRdmShape(),
        d_one_rdm,
        iodata.GetOneRdmShape()
    ));
    thrust::device_vector<double> all_ones(sizeof(double) * n_basis, 1.0);
    double *d_all_ones = thrust::raw_pointer_cast(all_ones.data());
    double *d_rho_hess_all;      // hessian (6, N) (row-order) upper-triangle order: xx, xy, xz, yy, yz, zz
    CUDA_CHECK(cudaMalloc((double **) &d_rho_hess_all, sizeof(double) * 6 * chunks.pts_per_iter));
    double *d_AOs_all ;
    CUDA_CHECK(cudaMalloc((double **) &d_AOs_all, sizeof(double) * n_basis * chunks.pts_per_iter));
    double *d_AOs_hess_all;
    CUDA_CHECK(cudaMalloc((double **) &d_AOs_hess_all, sizeof(double) * 6 * n_basis * chunks.pts_per_iter));
    // Allocate memory to hold the matrix-multiplcation between d_one_rdm and contraction
    double *d_MOs_all;
    CUDA_CHECK(cudaMalloc((double **) &d_MOs_all, sizeof(double) * chunks.pts_per_iter * n_basis));
    
    // Create the temporary pointers
    double *d_rho_hess = d_rho_hess_all, *d_AOs = d_AOs_all, *d_AOs_hess = d_AOs_hess_all;
    double *d_MOs = d_MOs_all;
    
    // Iterate through each chunk 
    size_t index_to_copy = 0;  
    size_t i_iter        = 0;
    
    while(index_to_copy < n_pts) {
        size_t npts_iter = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );

        // If it is the last iteration, I'll need to move the pointers to fit new size
        if (npts_iter != chunks.pts_per_iter) {
            d_rho_hess = d_rho_hess + 6 * (chunks.pts_per_iter - npts_iter);
            d_AOs = d_AOs + n_basis * (chunks.pts_per_iter - npts_iter);
            d_AOs_hess = d_AOs_hess + 6 * n_basis * (chunks.pts_per_iter - npts_iter);
            d_MOs = d_MOs + n_basis * (chunks.pts_per_iter - npts_iter);
        }
        
        // Allocate points and copy grid points column-order
        double *d_pts;
        CUDA_CHECK(cudaMalloc((double **) &d_pts, sizeof(double) * 3 * npts_iter));
        #pragma unroll 3
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpy(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Split up d_AOs_hess for efficient memory allocation
        double *d_AOs_deriv = &d_AOs_hess[3 * n_basis * npts_iter];
        cudaMemset(d_AOs_deriv, 0.0, sizeof(double) * 3 * n_basis * npts_iter);  // Reset to zero
        double *d_MOs_deriv = &d_AOs_hess[2 * n_basis * npts_iter];
        double *d_copy_arr =  &d_AOs_hess[    n_basis * npts_iter];
        std::vector<double*> d_pointers = {d_AOs_deriv, d_MOs_deriv, d_copy_arr};
        
        /**
         * Compute first term of the Hessian:
         *
         *  \sum_{kl} c_{kl} \frac{\partial \psi_k}{\partial x_n} \frac{\partial \psi_l}[\partial x_m}
         */
        cudaMemset(d_rho_hess, 0.0, sizeof(double) * 6 * npts_iter);  // Reset to zero
        evaluate_first_term(
            handle,
            molbasis,
            d_rho_hess,
            d_pts,
            d_one_rdm,
            d_all_ones,
            d_pointers,
            npts_iter,
            n_basis
        );
    
        /**
         * Compute second term of the Hessian:
         *
         *  \sum_{kl} c_{kl} \psi_k \frac{\partial^2 \psi_l}[\partial x_n \partial  x_m}
         */
         
        // Calculate the second derivatives of the contractions (6, M, N)
        cudaMemset(d_AOs_hess, 0.0, sizeof(double) * 6 * n_basis * npts_iter);  // Reset to zero
        dim3 threadsPerBlock(128);
        dim3 grid((npts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho_hess",
            d_AOs_hess,
            d_pts,
            npts_iter,
            n_basis,
            threadsPerBlock,
            grid
        );
        
        // Calculate the contractions (M, N) (row-major order)
        cudaMemset(d_AOs, 0.0, sizeof(double) * n_basis * npts_iter);  // Reset to zero
        dim3 threadsPerBlock2(128);
        dim3 grid2((npts_iter + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho",
            d_AOs,
            d_pts,
            npts_iter,
            n_basis,
            threadsPerBlock2,
            grid2
        );
        
        // Matrix multiple one-rdm with the contractions
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 npts_iter, n_basis, n_basis,
                                 &alpha, d_AOs, npts_iter,
                                 d_one_rdm, n_basis, &beta,
                                 d_MOs, npts_iter));
        
        // Create intermediate arrays to re-use memory.
        double* d_final = &d_pts[2 * npts_iter];
    
        int i_sec_deriv = 0;
        for(int i_deriv = 0; i_deriv < 3; i_deriv++) {
          for(int j_deriv = i_deriv; j_deriv < 3; j_deriv++) {
            // Get the second derivative: `i_deriv`th and `j_deriv`th derivative.
            double* d_sec_deriv = &d_AOs_hess[i_sec_deriv * npts_iter * n_basis];
            
            // Do a hadamard product with the original MOs and second derivs
            dim3 threadsPerBlock2(320);
            dim3 grid2((npts_iter * n_basis + threadsPerBlock.x - 1) / (threadsPerBlock.x));
            hadamard_product_outplace<<<grid2, threadsPerBlock2>>>(
                  d_AOs, d_sec_deriv, d_MOs, n_basis * npts_iter
            );
    
            // Take the sum to get the ith derivative of the electron density.
            //    This is done via matrix-vector multiplication of ones
            CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, npts_iter, n_basis,
                                     &alpha, d_AOs, npts_iter,
                                     d_all_ones, 1, &beta,
                                     d_final, 1));
    
            // Sum to device Hessian (column-order)
            sum_two_arrays_inplace<<<grid2, threadsPerBlock2>>>(
                &d_rho_hess[i_sec_deriv * npts_iter], d_final, npts_iter
            );
    
            i_sec_deriv += 1;  // Update to move to the next second derivative
          }
        }
    
        cudaFree(d_pts);
        
        // Multiply the derivative by two since electron density = sum | mo-contractions |^2
        dim3 threadsPerBlock3(320);
        dim3 grid3((6 * npts_iter + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
        multiply_scalar<<< grid3, threadsPerBlock3>>>(d_rho_hess, 2.0, 6 * npts_iter);
        
        // Return column implies iteration through points goes first then derivatives
        // Transfer the xx of device memory to host memory in row-major order.
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[index_to_copy],
                              d_rho_hess,
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the xy-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[n_pts + index_to_copy],
                              &d_rho_hess[npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the xz-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[2 * n_pts + index_to_copy],
                              &d_rho_hess[2 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the yx-coordinate from xy-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[3 * n_pts + index_to_copy],
                              &d_rho_hess[npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the yy-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[4 * n_pts + index_to_copy],
                              &d_rho_hess[3 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the yz-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[5 * n_pts + index_to_copy],
                              &d_rho_hess[4 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the zx-coordinate from xz-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[6 * n_pts + index_to_copy],
                              &d_rho_hess[2 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the zy-coordinate from yz-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[7 * n_pts + index_to_copy],
                              &d_rho_hess[4 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        // Transfer the zz-coordinate
        CUDA_CHECK(cudaMemcpy(&h_rho_hess[8 * n_pts + index_to_copy],
                              &d_rho_hess[5 * npts_iter],
                              sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
    
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter += 1;  // Update the index for each iteration.
    
    } // end while loop
    cudaFree(d_rho_hess_all);
    cudaFree(d_one_rdm);
    all_ones.clear();
    all_ones.shrink_to_fit();
    cudaFree(d_AOs_all);
    cudaFree(d_AOs_hess_all);
    cudaFree(d_MOs_all);

    // Cheap slow (but easier to code) method to return the row-major order
    if (return_row) {
        // Currently h_hess_electron_density is in column -order (N, 3, 3) so points first, then xx, xy, xz, yx, etc
        // mapping col-order (numbpts, deirv x_i, deriv x_j) -> global index: (n, i, j) = n + 3iN + jN
        // mapping (n, i, j) = 9n + 3i + j
        std::vector<double> h_hess_electron_density_row(9 * n_pts);
        for(size_t n = 0; n < n_pts; n++) {
          #pragma unroll 3
          for(size_t i = 0; i < 3; i++) {
            #pragma unroll 3
            for(size_t j = 0; j < 3; j++) {
              h_hess_electron_density_row[9 * n + 3 * i + j] =
                  h_rho_hess[n + 3 * i * n_pts + j * n_pts];
            }
          }
        }
        return h_hess_electron_density_row;
    }

    return h_rho_hess;
}