#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "eval_rho_grad.cuh"
#include "cuda_utils.cuh"
#include "cuda_basis_utils.cuh"
#include "basis_to_gpu.cuh"
#include "eval.cuh"

using namespace chemtools;

__device__ __forceinline__ void chemtools::eval_AOs_deriv(
          double*  d_AO_derivs,
    const double3  pt,
    const int      n_pts,
    const int      n_cshells,
          uint&    idx,
    const int      iorb_start
) {
    // Setup the initial variables.
    uint ibasis         = 0;                                        // Index to go over constant memory.
    uint iorb           = iorb_start;                               // Index to go over rows of d_contractions_array
    uint n_cshells_todo = (uint) g_constant_basis[ibasis++];        // Number of contracted shells in memory
    
    #pragma unroll 1
    for(int i_shell = 0; i_shell < n_cshells_todo; i_shell++) {
      double3 r_A = {
        pt.x - g_constant_basis[ibasis++],
        pt.y - g_constant_basis[ibasis++],
        pt.z - g_constant_basis[ibasis++]
      };

      uint n_seg_shells = (uint) g_constant_basis[ibasis++];
      uint n_prims      = (uint) g_constant_basis[ibasis++];
      #pragma unroll 1
      for(int iseg = 0; iseg < n_seg_shells; iseg++) {
        int L = (int) g_constant_basis[ibasis + n_prims + (n_prims + 1) * iseg];
        #pragma unroll 1
        for(int i_prim=0; i_prim < n_prims; i_prim++) {
          double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
          double a  = g_constant_basis[ibasis + i_prim];
          double ce = c * exp(- a * ( r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));

          // If S, P, D or F orbital/
          if(L == S_TYPE) {
            // Calculate the first derivative, then second, then third
            // d_contractions_array is stored as shape (3, M, N) where 3 is the three derivatives, M number of
            // contractions and N is the number of points, this conversion in row-major order is
            // N(M i_x + i_y) + i_z, where i_x=0,1,2,   i_y=0,...,M-1,   i_z=0,...,N-1.
            d_AO_derivs[n_pts * iorb + idx] +=
                normalization_primitive_s(a) *
                    (-2.0 * a * r_A.x) *  // d e^{-a x^2} / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                normalization_primitive_s(a) *
                    (-2.0 * a * r_A.y) *  // d e^{-a y^2} / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                normalization_primitive_s(a) *
                    (-2.0 * a * r_A.z) *  // d e^{-a z^2} / dz
                    ce;
          }
          else if (L == P_TYPE) {
            // First, second and third derivative of x_A e^{-a r_A^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                normalization_primitive_p(a) *
                    (1.0 - 2.0 * a * r_A.x * r_A.x) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                normalization_primitive_p(a) *
                    r_A.x * (-2.0 * a * r_A.y) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                normalization_primitive_p(a) *
                    r_A.x * (-2.0 * a * r_A.z) *
                    ce;
            // First, second and third derivative of y_A e^{-a r_A^2}
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                normalization_primitive_p(a) *
                    r_A.y * (-2.0 * a * r_A.x) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + (iorb + 1)) + idx] +=
                normalization_primitive_p(a) *
                    (1.0 - 2.0 * a * r_A.y * r_A.y) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + (iorb + 1)) + idx] +=
                normalization_primitive_p(a) *
                    r_A.y * (-2.0 * a * r_A.z) *
                    ce;
            // First, second and third derivative of z_A e^{-a r_A^2}
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                normalization_primitive_p(a) *
                    r_A.z * (-2.0 * a * r_A.x) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + (iorb + 2)) + idx] +=
                normalization_primitive_p(a) *
                    r_A.z * (-2.0 * a * r_A.y) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + (iorb + 2)) + idx] +=
                normalization_primitive_p(a) *
                    (1.0 - 2.0 * a * r_A.z * r_A.z) *
                    ce;
          }
          else if (L == D_TYPE) {
            // The ordering is ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
            // Take the first, second, third derivative of x_a^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                normalization_primitive_d(a, 2, 0, 0) *
                    r_A.x * (2.0 - 2.0 * a * r_A.x * r_A.x) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                normalization_primitive_d(a, 2, 0, 0) *
                    r_A.x * r_A.x * (-2.0 * a * r_A.y) *  // x_a^2 deriv e^{-a r^2} / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                normalization_primitive_d(a, 2, 0, 0) *
                    r_A.x * r_A.x * (-2.0 * a * r_A.z) *  // x_a^2 deriv e^{-a r^2} / dz
                    ce;
            // Take the first, second, third derivative of y_a^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                normalization_primitive_d(a, 0, 2, 0) *
                    r_A.y * r_A.y * (-2.0 * a * r_A.x) *   // deriv e^{-a x^2} / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                normalization_primitive_d(a, 0, 2, 0) *
                    r_A.y * (2.0 - 2.0 * a * r_A.y * r_A.y) *  // deriv y_A^2 e^{-a y^2} / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                normalization_primitive_d(a, 0, 2, 0) *
                    r_A.y * r_A.y * (-2.0 * a * r_A.z) *  // deriv e^{-a z^2} / dz
                    ce;
            // Take the first, second, third derivative of z_a^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                normalization_primitive_d(a, 0, 0, 2) *
                    r_A.z * r_A.z * (-2.0 * a * r_A.x) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                normalization_primitive_d(a, 0, 0, 2) *
                    r_A.z * r_A.z * (-2.0 * a * r_A.y) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                normalization_primitive_d(a, 0, 0, 2) *
                    r_A.z * (2.0 - 2.0 * a * r_A.z * r_A.z) *
                    ce;
            // Take the first, second, third derivative of x_a y_a e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                normalization_primitive_d(a, 1, 1, 0) *
                    (1.0 - 2.0 * a * r_A.x * r_A.x) * r_A.y *   // deriv x_a e^{-x} /dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                normalization_primitive_d(a, 1, 1, 0) *
                    r_A.x * (1.0 - 2.0 * a * r_A.y * r_A.y) *  // deriv y_a e^{-y} /dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                normalization_primitive_d(a, 1, 1, 0) *
                    r_A.x * r_A.y * (-2.0 * a * r_A.z) *  // deriv e^{-z} /dz
                    ce;
            // Take the first, second, third derivative of x_a z_a e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                normalization_primitive_d(a, 1, 0, 1) *
                    (1.0 - 2.0 * a * r_A.x * r_A.x) * r_A.z *  // deriv x_a e^{-a x^2} / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                normalization_primitive_d(a, 1, 0, 1) *
                    r_A.x * r_A.z * (-2.0 * a * r_A.y) *   // deriv e^{-a y^2} / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                normalization_primitive_d(a, 1, 0, 1) *
                    r_A.x * (1.0 - 2.0 * a * r_A.z * r_A.z) *  // deriv z_a e^{-a z^2} / dz
                    ce;
            // Take the first, second, third derivative of y_a z_a e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 5) + idx] +=
                normalization_primitive_d(a, 0, 1, 1) *
                    r_A.y * r_A.z * (-2.0 * a * r_A.x) *   // deriv e^{-a r^2} / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 5) + idx] +=
                normalization_primitive_d(a, 0, 1, 1) *
                    (1.0 - 2.0 * a * r_A.y * r_A.y) * r_A.z *  // deriv y_a e^{-a r^2} / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 5) + idx] +=
                normalization_primitive_d(a, 0, 1, 1) *
                    r_A.y * (1.0 - 2.0 * a * r_A.z * r_A.z) *  // deriv z_a e^{-a r^2} / dz
                    ce;
          }
          else if (L == DP_TYPE) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2'] which is m=[0, 1, -1, 2, -2]
            double norm_const = normalization_primitive_pure_d(a);
            // (x, y, z) derivative of ((2 z_A^2 - x_A^2 - y_A^2) / 2.0) e^{-a r^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                norm_const *
                    r_A.x * (-1 - a * ( 2.0 * r_A.z * r_A.z - r_A.x * r_A.x - r_A.y * r_A.y)) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                norm_const *
                    r_A.y * (-1 - a * ( 2.0 * r_A.z * r_A.z - r_A.x * r_A.x - r_A.y * r_A.y)) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                norm_const *
                    r_A.z * (2.0 - a * ( 2.0 * r_A.z * r_A.z - r_A.x * r_A.x - r_A.y * r_A.y)) *
                    ce;
            // Derivataive of sqrt(3) x_A * z_A e^{-a r^2}
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                norm_const *
                    sqrt(3.0) * (1.0 - 2.0 * a * r_A.x * r_A.x) * r_A.z *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                norm_const *
                  sqrt(3.0) * r_A.x * r_A.z * (-2.0 * a * r_A.y) *
                  ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.x * (1.0 - 2.0 * a * r_A.z * r_A.z) *
                    ce;
            // Derivative of sqrt(3) y_A * z_A e^{-a r^2}
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                norm_const *
                  sqrt(3.0) * r_A.y * r_A.z * (-2.0 * a * r_A.x) *
                  ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                norm_const *
                    sqrt(3.0) * (1.0 - 2.0 * a * r_A.y * r_A.y) * r_A.z *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.y * (1.0 - 2.0 * a * r_A.z * r_A.z) *
                    ce;
            // Derivative of sqrt(3) (x_A^2 - y_A^2) / 2.0  e^{-a r^2}
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.x * (1.0 - a * (r_A.x * r_A.x - r_A.y * r_A.y)) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.y * (-1.0 - a * (r_A.x * r_A.x - r_A.y * r_A.y)) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                norm_const *
                    sqrt(3.0) * (r_A.x * r_A.x - r_A.y * r_A.y) * (-a * r_A.z) *
                    ce;;
            // (x, y, z) derivative respectively of sqrt(3) * x_A * y_A e^{-a r^2}
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                norm_const *
                    sqrt(3.0) * (1.0 - 2.0 * a * r_A.x * r_A.x) * r_A.y *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.x * (1.0 - 2.0 * a * r_A.y * r_A.y) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                norm_const *
                    sqrt(3.0) * r_A.x * r_A.y * (-2.0 * a * r_A.z) *
                    ce;;
          }
          else if (L == F_TYPE) {
            // The ordering is ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
            // Take the first, second, third derivative of x_a^3 e^{-a r_a^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                normalization_primitive_f(a, 3, 0, 0) *
                    r_A.x*r_A.x*(-2*a*r_A.x*r_A.x + 3) *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                normalization_primitive_f(a, 3, 0, 0) *
                    -2*a*(r_A.x * r_A.x * r_A.x)*r_A.y *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                normalization_primitive_f(a, 3, 0, 0) *
                    -2*a*(r_A.x * r_A.x * r_A.x)*r_A.z *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y_a^3  e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                normalization_primitive_f(a, 0, 3, 0) *
                    -2*a*r_A.x*(r_A.y * r_A.y * r_A.y) *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                normalization_primitive_f(a, 0, 3, 0) *
                    r_A.y*r_A.y*(-2*a*r_A.y*r_A.y + 3) *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                normalization_primitive_f(a, 0, 3, 0) *
                    -2*a*(r_A.y * r_A.y * r_A.y)*r_A.z *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of z_A^3  e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                normalization_primitive_f(a, 0, 0, 3) *
                    -2*a*r_A.x*(r_A.z * r_A.z * r_A.z) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                normalization_primitive_f(a, 0, 0, 3) *
                    -2*a*r_A.y*(r_A.z * r_A.z * r_A.z) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                normalization_primitive_f(a, 0, 0, 3) *
                    r_A.z*r_A.z*(-2*a*r_A.z*r_A.z + 3) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a y_a^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                normalization_primitive_f(a, 1, 2, 0) *
                    r_A.y*r_A.y*(-2*a*r_A.x*r_A.x + 1) *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                normalization_primitive_f(a, 1, 2, 0) *
                    2*r_A.x*r_A.y*(-a*r_A.y*r_A.y + 1) *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                normalization_primitive_f(a, 1, 2, 0) *
                    -2*a*r_A.x*r_A.y*r_A.y*r_A.z *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a^2 y e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 0) *
                    2*r_A.x*r_A.y*(-a*r_A.x*r_A.x + 1) *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 0) *
                    r_A.x*r_A.x*(-2*a*r_A.y*r_A.y + 1) *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 0) *
                    -2*a*r_A.x*r_A.x*r_A.y*r_A.z *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a^2 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 1) *
                    2*r_A.x*r_A.z*(-a*r_A.x*r_A.x + 1) *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 1) *
                    -2*a*r_A.x*r_A.x*r_A.y*r_A.z *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 1) *
                    r_A.x*r_A.x*(-2*a*r_A.z*r_A.z + 1) *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a z_a^2  e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 0, 2) *
                    r_A.z*r_A.z*(-2*a*r_A.x*r_A.x + 1) *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 0, 2) *
                    -2*a*r_A.x*r_A.y*r_A.z*r_A.z *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 0, 2) *
                    2*r_A.x*r_A.z*(-a*r_A.z*r_A.z + 1) *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y_a z_a^2  e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 7) + idx] +=
                normalization_primitive_f(a, 0, 1, 2) *
                    -2*a*r_A.x*r_A.y*r_A.z*r_A.z *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 7) + idx] +=
                normalization_primitive_f(a, 0, 1, 2) *
                    r_A.z*r_A.z*(-2*a*r_A.y*r_A.y + 1) *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 7) + idx] +=
                normalization_primitive_f(a, 0, 1, 2) *
                    2*r_A.y*r_A.z*(-a*r_A.z*r_A.z + 1) *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y_a^2 z_a  e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 8) + idx] +=
                normalization_primitive_f(a, 0, 2, 1) *
                    -2*a*r_A.x*r_A.y*r_A.y*r_A.z *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 8) + idx] +=
                normalization_primitive_f(a, 0, 2, 1) *
                    2*r_A.y*r_A.z*(-a*r_A.y*r_A.y + 1) *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 8) + idx] +=
                normalization_primitive_f(a, 0, 2, 1) *
                    r_A.y*r_A.y*(-2*a*r_A.z*r_A.z + 1) *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a y_a z_a e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 1, 1) *
                    r_A.y*r_A.z*(-2*a*r_A.x*r_A.x + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 1, 1) *
                    r_A.x*r_A.z*(-2*a*r_A.y*r_A.y + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 1, 1) *
                    r_A.x*r_A.y*(-2*a*r_A.z*r_A.z + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
          }
          else if (L == SF_TYPE) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
            // This was done using wolframalpha for the formulas
            double norm_const = normalization_primitive_pure_f(a);
            // (x, y, z) derivative of ((2 z_A^2 - 3 x_A^2 - 3 y_A^2) / 2.0) e^{-a r^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                norm_const *
                    r_A.x*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                norm_const *
                    r_A.y*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                norm_const *
                    (
                        (2*a*r_A.z*r_A.z*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 2*r_A.z*r_A.z) - 3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 6*r_A.z*r_A.z)/2
                    ) *
                    ce;
            // Derivataive of sqrt(1.5) (4 z_a^2 x_A   - x_A^3  - y_A^2 x_A ) / 2.0 e^{-a r^2} , m=1
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                norm_const *
                    sqrt(1.5) *
                    (
                        (2*a*r_A.x*r_A.x*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) - 3*r_A.x*r_A.x - r_A.y*r_A.y + 4*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                norm_const *
                    sqrt(1.5) * r_A.x*r_A.y*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) - 1) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                norm_const *
                    sqrt(1.5) * r_A.x*r_A.z*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) + 4) *
                    ce;
            // Derivative of  sqrt(1.5) (4 z_a^2 y_A   - x_A^2 y_A  - y_A^3 ) / 2.0 e^{-a r^2}  m = -1
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                norm_const *
                    sqrt(1.5) * (
                     r_A.x*r_A.y*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) - 1)
                     ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                norm_const *
                    sqrt(1.5) * (
                      (2*a*r_A.y*r_A.y*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) - r_A.x*r_A.x - 3*r_A.y*r_A.y + 4*r_A.z*r_A.z)/2
                     ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                norm_const *
                    sqrt(1.5) *  (
                    r_A.y*r_A.z*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 4*r_A.z*r_A.z) + 4)
                    ) *
                    ce;
            //  Derivative of  sqrt(15) (x_A^2  - y_A^2) z / 2.0 e^{-a r^2}  m = 2
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                norm_const *
                    sqrt(15.0) * r_A.x*r_A.z*(-a*(r_A.x*r_A.x - r_A.y*r_A.y) + 1) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                norm_const *
                    sqrt(15.0) * r_A.y*r_A.z*(-a*(r_A.x*r_A.x - r_A.y*r_A.y) - 1) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                norm_const *
                    sqrt(15.0) * (-2*a*r_A.z*r_A.z*(r_A.x*r_A.x - r_A.y*r_A.y) + r_A.x*r_A.x - r_A.y*r_A.y)/2 *
                    ce;
            //  Derivative of  sqrt(15) x y z         m = -2
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                norm_const *
                    sqrt(15.0) * r_A.y*r_A.z*(-2*a*r_A.x*r_A.x + 1) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                norm_const *
                    sqrt(15.0) * r_A.x*r_A.z*(-2*a*r_A.y*r_A.y + 1) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                norm_const *
                    sqrt(15.0) * r_A.x*r_A.y*(-2*a*r_A.z*r_A.z + 1) *
                    ce;
            //  Derivative of  sqrt(2.5) * (x^2 - 3 y^2) x / 2         m = 3
            d_AO_derivs[n_pts * (iorb + 5) + idx] +=
                norm_const *
                    sqrt(2.5) * (
                    (-2*a*r_A.x*r_A.x*(r_A.x*r_A.x - 3*r_A.y*r_A.y) + 3*r_A.x*r_A.x - 3*r_A.y*r_A.y)/2
                )  *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 5) + idx] +=
                norm_const *
                    sqrt(2.5) * r_A.x*r_A.y*(-a*(r_A.x*r_A.x - 3*r_A.y*r_A.y) - 3) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 5) + idx] +=
                norm_const *
                    sqrt(2.5) * -a*r_A.x*r_A.z*(r_A.x*r_A.x - 3*r_A.y*r_A.y) *
                    ce;
            //  Derivative of  sqrt(2.5) (3x^2 - y^2) y / 2        m = -3
            d_AO_derivs[n_pts * (iorb + 6) + idx] +=
                norm_const *
                    sqrt(2.5) * r_A.x*r_A.y*(-a*(3*r_A.x*r_A.x - r_A.y*r_A.y) + 3) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 6) + idx] +=
                norm_const *
                    sqrt(2.5) * (
                    (-2*a*r_A.y*r_A.y*(3*r_A.x*r_A.x - r_A.y*r_A.y) + 3*r_A.x*r_A.x - 3*r_A.y*r_A.y)/2
                ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 6) + idx] +=
                norm_const *
                    sqrt(2.5) * -a*r_A.y*r_A.z*(3*r_A.x*r_A.x - r_A.y*r_A.y) *
                    ce;
          }
          else if (L == G_TYPE) {
            // The ordering is ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy', 'xxzz', 'xxyz',
            //                    'xxyy', 'xxxz', 'xxxy', 'xxxx']
            // Take the first, second, third derivative of z^4 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 14) + idx] +=
                normalization_primitive_f(a, 0, 0, 4) *
                    -2*a*r_A.x*(r_A.z * r_A.z * r_A.z * r_A.z) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 14) + idx] +=
                normalization_primitive_f(a, 0, 0, 4) *
                    -2*a*r_A.y*(r_A.z * r_A.z * r_A.z * r_A.z) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 14) + idx] +=
                normalization_primitive_f(a, 0, 0, 4) *
                    2*(r_A.z * r_A.z * r_A.z)*(-a*r_A.z*r_A.z + 2) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y z^3 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 13) + idx] +=
                normalization_primitive_f(a, 0, 1, 3) *
                    -2*a*r_A.x*r_A.y*(r_A.z * r_A.z * r_A.z) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 13) + idx] +=
                normalization_primitive_f(a, 0, 1, 3) *
                    (r_A.z * r_A.z * r_A.z)*(-2*a*r_A.y*r_A.y + 1) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 13) + idx] +=
                normalization_primitive_f(a, 0, 1, 3) *
                    r_A.y*r_A.z*r_A.z*(-2*a*r_A.z*r_A.z + 3) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y^2 z^2 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 12) + idx] +=
                normalization_primitive_f(a, 0, 2, 2) *
                    -2*a*r_A.x*r_A.y*r_A.y*r_A.z*r_A.z *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 12) + idx] +=
                normalization_primitive_f(a, 0, 2, 2) *
                    2*r_A.y*r_A.z*r_A.z*(-a*r_A.y*r_A.y + 1) *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 12) + idx] +=
                normalization_primitive_f(a, 0, 2, 2) *
                    2*r_A.y*r_A.y*r_A.z*(-a*r_A.z*r_A.z + 1) *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y^3 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 11) + idx] +=
                normalization_primitive_f(a, 0, 3, 1) *
                    -2*a*r_A.x*(r_A.y * r_A.y * r_A.y)*r_A.z *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 11) + idx] +=
                normalization_primitive_f(a, 0, 3, 1) *
                    r_A.y*r_A.y*r_A.z*(-2*a*r_A.y*r_A.y + 3) *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 11) + idx] +=
                normalization_primitive_f(a, 0, 3, 1) *
                    (r_A.y * r_A.y * r_A.y)*(-2*a*r_A.z*r_A.z + 1) *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of y^4 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 10) + idx] +=
                normalization_primitive_f(a, 0, 4, 0) *
                    -2*a*r_A.x*(r_A.y * r_A.y * r_A.y * r_A.y) *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 10) + idx] +=
                normalization_primitive_f(a, 0, 4, 0) *
                    2*(r_A.y * r_A.y * r_A.y)*(-a*r_A.y*r_A.y + 2) *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 10) + idx] +=
                normalization_primitive_f(a, 0, 4, 0) *
                    -2*a*(r_A.y * r_A.y * r_A.y * r_A.y)*r_A.z *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x z^3 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 0, 3) *
                    (r_A.z * r_A.z * r_A.z)*(-2*a*r_A.x*r_A.x + 1) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 0, 3) *
                    -2*a*r_A.x*r_A.y*(r_A.z * r_A.z * r_A.z) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 9) + idx] +=
                normalization_primitive_f(a, 1, 0, 3) *
                    r_A.x*r_A.z*r_A.z*(-2*a*r_A.z*r_A.z + 3) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x y z^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 8) + idx] +=
                normalization_primitive_f(a, 1, 1, 2) *
                    r_A.y*r_A.z*r_A.z*(-2*a*r_A.x*r_A.x + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 8) + idx] +=
                normalization_primitive_f(a, 1, 1, 2) *
                    r_A.x*r_A.z*r_A.z*(-2*a*r_A.y*r_A.y + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 8) + idx] +=
                normalization_primitive_f(a, 1, 1, 2) *
                    2*r_A.x*r_A.y*r_A.z*(-a*r_A.z*r_A.z + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x y^2 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 7) + idx] +=
                normalization_primitive_f(a, 1, 2, 1) *
                    r_A.y*r_A.y*r_A.z*(-2*a*r_A.x*r_A.x + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 7) + idx] +=
                normalization_primitive_f(a, 1, 2, 1) *
                    2*r_A.x*r_A.y*r_A.z*(-a*r_A.y*r_A.y + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 7) + idx] +=
                normalization_primitive_f(a, 1, 2, 1) *
                    r_A.x*r_A.y*r_A.y*(-2*a*r_A.z*r_A.z + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x y^3 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 3, 0) *
                    (r_A.y * r_A.y * r_A.y)*(-2*a*r_A.x*r_A.x + 1) *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 3, 0) *
                    r_A.x*r_A.y*r_A.y*(-2*a*r_A.y*r_A.y + 3) *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 6) + idx] +=
                normalization_primitive_f(a, 1, 3, 0) *
                    -2*a*r_A.x*(r_A.y * r_A.y * r_A.y)*r_A.z *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x^2 z^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 2) *
                    2*r_A.x*r_A.z*r_A.z*(-a*r_A.x*r_A.x + 1) *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 2) *
                    -2*a*r_A.x*r_A.x*r_A.y*r_A.z*r_A.z *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 5) + idx] +=
                normalization_primitive_f(a, 2, 0, 2) *
                    2*r_A.x*r_A.x*r_A.z*(-a*r_A.z*r_A.z + 1) *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x^2 yz e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 1) *
                    2*r_A.x*r_A.y*r_A.z*(-a*r_A.x*r_A.x + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 1) *
                    r_A.x*r_A.x*r_A.z*(-2*a*r_A.y*r_A.y + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                normalization_primitive_f(a, 2, 1, 1) *
                    r_A.x*r_A.x*r_A.y*(-2*a*r_A.z*r_A.z + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x^2 y^2 e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                normalization_primitive_f(a, 2, 2, 0) *
                    2*r_A.x*r_A.y*r_A.y*(-a*r_A.x*r_A.x + 1) *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                normalization_primitive_f(a, 2, 2, 0) *
                    2*r_A.x*r_A.x*r_A.y*(-a*r_A.y*r_A.y + 1) *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                normalization_primitive_f(a, 2, 2, 0) *
                    -2*a*r_A.x*r_A.x*r_A.y*r_A.y*r_A.z *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x^3 z e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                normalization_primitive_f(a, 3, 0, 1) *
                    r_A.x*r_A.x*r_A.z*(-2*a*r_A.x*r_A.x + 3) *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                normalization_primitive_f(a, 3, 0, 1) *
                    -2*a*(r_A.x * r_A.x * r_A.x)*r_A.y*r_A.z *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                normalization_primitive_f(a, 3, 0, 1) *
                    (r_A.x * r_A.x * r_A.x)*(-2*a*r_A.z*r_A.z + 1) *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x^3 y e^{-a r_a^2}
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                normalization_primitive_f(a, 3, 1, 0) *
                    r_A.x*r_A.x*r_A.y*(-2*a*r_A.x*r_A.x + 3) *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                normalization_primitive_f(a, 3, 1, 0) *
                    (r_A.x * r_A.x * r_A.x)*(-2*a*r_A.y*r_A.y + 1) *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                normalization_primitive_f(a, 3, 1, 0) *
                    -2*a*(r_A.x * r_A.x * r_A.x)*r_A.y*r_A.z *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
            // Take the first, second, third derivative of x_a^4 e^{-a r_a^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                normalization_primitive_f(a, 4, 0, 0) *
                    2*(r_A.x * r_A.x * r_A.x)*(-a*r_A.x*r_A.x + 2) *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                normalization_primitive_f(a, 4, 0, 0) *
                    -2*a*(r_A.x * r_A.x * r_A.x * r_A.x)*r_A.y *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                normalization_primitive_f(a, 4, 0, 0) *
                    
                    -2*a*(r_A.x * r_A.x * r_A.x * r_A.x)*r_A.z *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    ce;
          }
          else if (L == SG_TYPE) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
            // This was done using wolframalpha for the formulas
            double norm_const = normalization_primitive_pure_g(a);
            // (x, y, z) derivative of e^{-a r^2}
            d_AO_derivs[n_pts * iorb + idx] +=
                norm_const *
                    (
                        r_A.x*(-a*(35*(r_A.z * r_A.z * r_A.z * r_A.z) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) +
                        3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) * (r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) + 6*r_A.x*r_A.x + 6*r_A.y*r_A.y -
                        24*r_A.z*r_A.z)/4
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb) + idx] +=
                norm_const *
                    (
                        r_A.y*(-a*(35*(r_A.z * r_A.z * r_A.z * r_A.z) - 30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) +
                        3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) * (r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) +
                        6*r_A.x*r_A.x + 6*r_A.y*r_A.y - 24*r_A.z*r_A.z)/4
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb) + idx] +=
                norm_const *
                    (
                        r_A.z*(-a*(35*(r_A.z * r_A.z * r_A.z * r_A.z) -
                        30*r_A.z*r_A.z*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) +
                        3*(r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z) * (r_A.x*r_A.x + r_A.y*r_A.y + r_A.z*r_A.z)) -
                        24*r_A.x*r_A.x - 24*r_A.y*r_A.y + 16*r_A.z*r_A.z)/4
                    ) *
                    ce;
            // Derivataive of sqrt(2.5) (7z^2 - 3 (x^2 + y^2 + z^2)) xz / 2.0 , m=1
            d_AO_derivs[n_pts * (iorb + 1) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.z*(2*a*r_A.x*r_A.x*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) -
                        9*r_A.x*r_A.x - 3*r_A.y*r_A.y + 4*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 1) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.x*r_A.y*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) - 3)
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 1) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.x*(2*a*r_A.z*r_A.z*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) -
                        3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 12*r_A.z*r_A.z)/2
                    )*
                    ce;
            // Derivataive of sqrt(2.5) (7z^2 - 3 (x^2 + y^2 + z^2)) xz / 2.0 , m= -1
            d_AO_derivs[n_pts * (iorb + 2) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.x*r_A.y*r_A.z*(a*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) - 3)
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 2) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.z*(2*a*r_A.y*r_A.y*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) - 3*r_A.x*r_A.x - 9*r_A.y*r_A.y + 4*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 2) + idx] +=
                norm_const *
                    sqrt(2.5) *
                    (
                        r_A.y*(2*a*r_A.z*r_A.z*(3*r_A.x*r_A.x + 3*r_A.y*r_A.y - 4*r_A.z*r_A.z) -
                        3*r_A.x*r_A.x - 3*r_A.y*r_A.y + 12*r_A.z*r_A.z)/2
                    ) *
                    ce;
            // Derivative  of sqrt(5)  (7 * z**2 + r2) * (x**2 - y**2) * exp / 4,  m= 2
            d_AO_derivs[n_pts * (iorb + 3) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.x*(a*(r_A.x*r_A.x - r_A.y*r_A.y)*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) -
                        2*r_A.x*r_A.x + 6*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 3) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.y*(a*(r_A.x*r_A.x - r_A.y*r_A.y)*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) +
                        2*r_A.y*r_A.y - 6*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 3) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.z*(r_A.x*r_A.x - r_A.y*r_A.y)*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) + 6)/2
                    ) *
                    ce;
            // Derivataive of sqrt(5)     (7 * z**2 - r2) * x * y * exp / 2,        m= -2
            d_AO_derivs[n_pts * (iorb + 4) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.y*(2*a*r_A.x*r_A.x*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) -
                        3*r_A.x*r_A.x - r_A.y*r_A.y + 6*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 4) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.x*(2*a*r_A.y*r_A.y*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) -
                        r_A.x*r_A.x - 3*r_A.y*r_A.y + 6*r_A.z*r_A.z)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 4) + idx] +=
                norm_const *
                    sqrt(5.0) *
                    (
                        r_A.x*r_A.y*r_A.z*(a*(r_A.x*r_A.x + r_A.y*r_A.y - 6*r_A.z*r_A.z) + 6)
                    ) *
                    ce;
            // Derivataive of sqrt(35 / 2)      (x**2 - 3 * y**2) * x * z * exp / 2,       m= 3
            d_AO_derivs[n_pts * (iorb + 5) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.z*(-2*a*r_A.x*r_A.x*(r_A.x*r_A.x - 3*r_A.y*r_A.y) + 3*r_A.x*r_A.x - 3*r_A.y*r_A.y)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 5) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.x*r_A.y*r_A.z*(-a*(r_A.x*r_A.x - 3*r_A.y*r_A.y) - 3)
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 5) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.x*(r_A.x*r_A.x - 3*r_A.y*r_A.y)*(-2*a*r_A.z*r_A.z + 1)/2
                    ) *
                    ce;
            // Derivataive of sqrt(35 / 2)    (3 * x**2 - y**2) * y * z * exp / 2,       m= -3
            d_AO_derivs[n_pts * (iorb + 6) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.x*r_A.y*r_A.z*(-a*(3*r_A.x*r_A.x - r_A.y*r_A.y) + 3)
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 6) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.z*(-2*a*r_A.y*r_A.y*(3*r_A.x*r_A.x - r_A.y*r_A.y) + 3*r_A.x*r_A.x - 3*r_A.y*r_A.y)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 6) + idx] +=
                norm_const *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A.y*(3*r_A.x*r_A.x - r_A.y*r_A.y)*(-2*a*r_A.z*r_A.z + 1)/2
                    ) *
                    ce;
            // Derivataive of sqrt(35)     (x**4 - 6 * x**2 * y**2 + y**4) * exp / 8,,       m= 4
            d_AO_derivs[n_pts * (iorb + 7) + idx] +=
                norm_const *
                    sqrt(35.0) *
                    (
                        r_A.x*(-a*((r_A.x * r_A.x * r_A.x * r_A.x) - 6*r_A.x*r_A.x*r_A.y*r_A.y + (r_A.y * r_A.y * r_A.y * r_A.y)) + 2*r_A.x*r_A.x -
                        6*r_A.y*r_A.y)/4
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 7) + idx] +=
                norm_const *
                    sqrt(35.0) *
                    (
                        r_A.y*(-a*((r_A.x * r_A.x * r_A.x * r_A.x) - 6*r_A.x*r_A.x*r_A.y*r_A.y + (r_A.y * r_A.y * r_A.y * r_A.y)) - 6*r_A.x*r_A.x +
                        2*r_A.y*r_A.y)/4
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 7) + idx] +=
                norm_const *
                    sqrt(35) *
                    (
                        -a*r_A.z*((r_A.x * r_A.x * r_A.x * r_A.x) - 6*r_A.x*r_A.x*r_A.y*r_A.y + (r_A.y * r_A.y * r_A.y * r_A.y))/4
                    ) *
                    ce;
            // Derivataive of sqrt(35)      (x**2 - y**2) * x * y * exp / 2 ,       m= -4
            d_AO_derivs[n_pts * (iorb + 8) + idx] +=
                norm_const *
                    sqrt(35) *
                    (
                        r_A.y*(-2*a*r_A.x*r_A.x*(r_A.x*r_A.x - r_A.y*r_A.y) + 3*r_A.x*r_A.x - r_A.y*r_A.y)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells + iorb + 8) + idx] +=
                norm_const *
                    sqrt(35) *
                    (
                        r_A.x*(-2*a*r_A.y*r_A.y*(r_A.x*r_A.x - r_A.y*r_A.y) + r_A.x*r_A.x - 3*r_A.y*r_A.y)/2
                    ) *
                    ce;
              d_AO_derivs[n_pts * (n_cshells * 2 + iorb + 8) + idx] +=
                norm_const *
                    sqrt(35) *
                    (
                        -a*r_A.x*r_A.y*r_A.z*(r_A.x*r_A.x - r_A.y*r_A.y)
                    ) *
                    ce;
          }// End angmoms.
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


__global__ __launch_bounds__(128) void chemtools::eval_AOs_derivs_on_any_grid(
          double* __restrict__ d_AO_derivs,
    const double* __restrict__ d_points,
    const int     n_pts,
    const int     n_cshells,
    const int     iorb_start
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pts) {
        // Get the grid points where `d_points` is in column-major order with shape (N, 3)
        double pt_x = d_points[idx];
        double pt_y = d_points[idx + n_pts];
        double pt_z = d_points[idx + n_pts * 2];
        // Evaluate the contraction derivatives and store it in d_AO_vals
        chemtools::eval_AOs_deriv(d_AO_derivs, {pt_x, pt_y, pt_z}, n_pts, n_cshells, idx, iorb_start);
    }
}


__host__ std::vector<double> chemtools::evaluate_contraction_derivatives(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
) {
    // Get the molecular basis from iodata and put it in constant memory of the gpu.
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();

    // The output of the contractions in column-major order with shape (3, M, N).
    std::vector<double> h_contractions(3 * n_basis * knumb_points);
    
    // Transfer grid points to GPU, this is in column order with shape (N, 3)
    double* d_points;
    cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
    cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));
    
    // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
    double* d_deriv_contractions;
    cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions, sizeof(double) * 3 * knumb_points * n_basis));
    dim3 threadsPerBlock(128);
    dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    evaluate_scalar_quantity_density(
        molbasis,
        false,
        false,
        "rho_deriv",
        d_deriv_contractions,
        d_points,
        knumb_points,
        n_basis,
        threadsPerBlock,
        grid
    );

    // Transfer from device memory to host memory
    cuda_check_errors(cudaMemcpy(&h_contractions[0],
                                       d_deriv_contractions,
                                       sizeof(double) * 3 * knumb_points * n_basis, cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_deriv_contractions);
    
    return h_contractions;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_gradient(
    IOData& iodata, const double* h_points, const int knumb_points, const bool return_row
) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  std::vector<double> gradient = evaluate_electron_density_gradient_handle(
      handle, iodata, h_points, knumb_points, return_row
  );
  CUBLAS_CHECK(cublasDestroy(handle));
  return gradient;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_gradient_handle(
          cublasHandle_t& handle,
          IOData&         iodata,
    const double*         h_points,
    const int             n_pts,
    const bool            return_row
) {
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    //  Solve for N (5 * N M + M^2 + 3N) * 8 bytes = Free memory (in bytes)
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [return_row](size_t mem, size_t numb_basis){
            // Return_row has addition 3N, so 3->6
          return return_row ? ((mem / sizeof(double))  - numb_basis - numb_basis * numb_basis) / (5 * numb_basis + 6):
          ((mem / sizeof(double))  - numb_basis - numb_basis * numb_basis) / (5 * numb_basis + 3);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Resulting electron density gradient
    std::vector<double> h_grad_rho(3 * n_pts);
    
    // Allocate All Device Variables, It is actually faster than doing per iteration
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
    double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
    double *d_pts_all = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_pts_all, sizeof(double) * 3 * chunks.pts_per_iter));
    // Allocate device memory for contractions row-major (M, N)
    double *d_AOs_all = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_AOs_all, sizeof(double) * n_basis * chunks.pts_per_iter));
    // Temp array matrix multiplication of one_rdm and contraction array.
    double *d_temp_all = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_temp_all, sizeof(double) * n_basis * chunks.pts_per_iter));
    double *d_AOs_deriv_all = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_AOs_deriv_all, sizeof(double) * 3 * chunks.pts_per_iter * n_basis));
    
    // Create temporary points so that it is easy to update the last iteration
    double *d_pts = d_pts_all, *d_AOs = d_AOs_all;
    double *d_temp = d_temp_all, *d_AOs_deriv = d_AOs_deriv_all;
    
    // Iterate through every chunk of the row
    size_t index_to_copy = 0;
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        size_t npts_iter = std::min(
              n_pts - i_iter * chunks.pts_per_iter,
              chunks.pts_per_iter
        );
        // If it is the last iteration, I'll need to move the pointers to fit new size
        if (npts_iter != chunks.pts_per_iter) {
            d_pts = d_pts + 3 * (chunks.pts_per_iter - npts_iter);
            d_AOs = d_AOs + n_basis * (chunks.pts_per_iter - npts_iter);
            d_temp = d_temp + n_basis * (chunks.pts_per_iter - npts_iter);
            d_AOs_deriv = d_AOs_deriv + 3 * n_basis * (chunks.pts_per_iter - npts_iter);
        }
        
        // Set Zero to the atomic and derivative orbitals.
        if (i_iter > 0) {
            cudaMemsetAsync(d_AOs, 0.0,       sizeof(double) * n_basis * npts_iter);
            cudaMemsetAsync(d_AOs_deriv, 0.0, sizeof(double) * 3 * n_basis * npts_iter);
        }
        
        // Allocate points and copy grid points column-order
        #pragma unroll
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpyAsync(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Evaluate Atomic Orbitals
        constexpr int THREADS_PER_BLOCK = 128;
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks((npts_iter + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho",
            d_AOs,
            d_pts,
            npts_iter,
            n_basis,
            threads,
            blocks
        );
        
        // Matrix mult. of one rdm with the contractions array. Everything is in row major order.
        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            npts_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
            &alpha,
            d_AOs, npts_iter,
            d_one_rdm, iodata.GetOneRdmShape(),
            &beta,
            d_temp, npts_iter
        ));
        
        // Evaluate derivatives of AOs (3, M, N) row-order
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho_deriv",
            d_AOs_deriv,
            d_pts,
            npts_iter,
            n_basis,
            threads,
            blocks
        );
        
        // Gradient electron density col-order (Re-Use Pointer)
        double *d_grad_rho = d_pts;
        
        // Hadamard d_temp and each derv in d_AOs_deriv and multiply by two since
        //   electron density = sum | mo-contractions |^2
        constexpr int HADAMARD_THREADS = 1024;
        dim3 hadamard_threads(HADAMARD_THREADS);
        dim3 hadamard_blocks(
            (npts_iter * n_basis + HADAMARD_THREADS - 1) / HADAMARD_THREADS
        );
        hadamard_product_tensor_with_mat_with_multiply_by_two<3><<<hadamard_blocks, hadamard_threads>>>(
            d_AOs_deriv, d_temp, n_basis * npts_iter
        );
        
        // For each derivative, calculate the derivative of electron density seperately.
        #pragma unroll 3
        for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
            // Ith deriv of AOs (M, N) row-order
            double *d_ith_deriv = &d_AOs_deriv[i_deriv * npts_iter * n_basis];
            
            // Summation of rows
            CUBLAS_CHECK(cublasDgemv(
                handle, CUBLAS_OP_N, npts_iter, n_basis,
                &alpha,
                d_ith_deriv, npts_iter, deviceVecPtr, 1,
                &beta,
                &d_grad_rho[i_deriv * npts_iter], 1)
            );
        }
        
        // Transfer from column-major to row-major order
        if(return_row){
            double *d_grad_clone;
            CUDA_CHECK(cudaMalloc((double **) &d_grad_clone, sizeof(double) * 3 * npts_iter));
            CUDA_CHECK(cudaMemcpy(
                d_grad_clone,
                d_grad_rho,
                sizeof(double) * 3 * npts_iter,
                cudaMemcpyDeviceToDevice
            ));
            const double alpha = 1.0;
            const double beta = 0.0;
            CUBLAS_CHECK(cublasDgeam(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                3, npts_iter,
                &alpha,
                d_grad_rho, npts_iter,
                &beta,
                d_grad_rho, 3, d_grad_clone, 3)
            );
        
            // Transfer the gradient of device memory to host memory in row-major order.
            CUDA_CHECK(cudaMemcpy(&h_grad_rho[3 * index_to_copy],
                                       d_grad_clone,
                                       sizeof(double) * 3 * npts_iter, cudaMemcpyDeviceToHost));
            cudaFree(d_grad_clone);
        }
        else {
            // Transfer the x-coordinate gradient of device memory to host memory in row-major order.
            CUDA_CHECK(cudaMemcpy(&h_grad_rho[index_to_copy],
                                       d_grad_rho,
                                       sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
            // Transfer the y-coordinate
            CUDA_CHECK(cudaMemcpy(&h_grad_rho[n_pts + index_to_copy],
                                               &d_grad_rho[npts_iter],
                                               sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
            // Transfer the z-coordinate
            CUDA_CHECK(cudaMemcpy(&h_grad_rho[2 * n_pts + index_to_copy],
                                               &d_grad_rho[2 * npts_iter],
                                               sizeof(double) * npts_iter, cudaMemcpyDeviceToHost));
        }
        
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter++;
    }
    
    // Deallocate all variables
    cudaFree(d_one_rdm);
    cudaFree(d_AOs_all);
    cudaFree(d_temp_all);
    cudaFree(d_AOs_deriv_all);
    cudaFree(d_pts_all);
    all_ones.clear();
    all_ones.shrink_to_fit();

  return h_grad_rho;
}
