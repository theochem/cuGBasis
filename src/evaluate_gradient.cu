#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "../include/evaluate_gradient.cuh"
#include "../include/evaluate_density.cuh"
#include "../include/contracted_shell.h"
#include "../include/cuda_utils.cuh"
#include "../include/cuda_basis_utils.cuh"
#include "../include/basis_to_gpu.cuh"


__device__ extern d_func_t chemtools::p_evaluate_deriv_contractions = chemtools::evaluate_derivatives_contractions_from_constant_memory;

__device__ void chemtools::evaluate_derivatives_contractions_from_constant_memory(
    double* d_deriv_contracs, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_contractions_start
) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;  // map thread index to the index of hte points
  if (global_index < knumb_points){
    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];

    // Setup the initial variables.
    int iconst = 0;                                                          // Index to go over constant memory.
    unsigned int icontractions = i_contractions_start;                       // Index to go over rows of d_contractions_array
    unsigned int numb_contracted_shells = (int) g_constant_basis[iconst++];
    #pragma unroll
    for(int icontr_shell = 0; icontr_shell < numb_contracted_shells; icontr_shell++) {
      // Distance from the grid point to the center of the contracted shell.
      double r_A_x = (grid_x - g_constant_basis[iconst++]);
      double r_A_y = (grid_y - g_constant_basis[iconst++]);
      double r_A_z = (grid_z - g_constant_basis[iconst++]);
//      double radius_sq = pow(r_A_x, 2.0) + pow(r_A_y, 2.0) + pow(r_A_z, 2.0);

      int numb_segment_shells = (int) g_constant_basis[iconst++];
      int numb_primitives = (int) g_constant_basis[iconst++];
      // iconst from here=H+0 to H+(numb_primitives - 1) is the exponents, need this to constantly reiterate them.
      for(int i_segm_shell=0; i_segm_shell < numb_segment_shells; i_segm_shell++) {
        // Add the number of exponents, then add extra coefficients to enumerate.
        int angmom = (int) g_constant_basis[iconst + numb_primitives + (numb_primitives + 1) * i_segm_shell];
        for(int i_prim=0; i_prim < numb_primitives; i_prim++) {
          double coeff_prim = g_constant_basis[iconst + numb_primitives * (i_segm_shell + 1) + i_prim + 1 + i_segm_shell];
          double alpha = g_constant_basis[iconst + i_prim];
          double exponential = exp(- alpha * ( r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z));

          // If S, P, D or F orbital/
          if(angmom == 0) {
            // Calculate the first derivative, then second, then third
            // d_contractions_array is stored as shape (3, M, N) where 3 is the three derivatives, M number of
            // contractions and N is the number of points, this conversion in row-major order is
            // N(M i_x + i_y) + i_z, where i_x=0,1,2,   i_y=0,...,M-1,   i_z=0,...,N-1.
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                chemtools::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_x) *  // d e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                chemtools::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_y) *  // d e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                chemtools::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_z) *  // d e^{-a z^2} / dz
                    exponential;
          }
          else if (angmom == 1) {
            // First, second and third derivative of x_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_x * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_x * (-2.0 * alpha * r_A_z) *
                    exponential;
            // First, second and third derivative of y_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_y * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + (icontractions + 1)) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_y * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + (icontractions + 1)) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_y * (-2.0 * alpha * r_A_z) *
                    exponential;
            // First, second and third derivative of z_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_z * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + (icontractions + 2)) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_z * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + (icontractions + 2)) + global_index] +=
                chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
          }
          else if (angmom == 2) {
            // The ordering is ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
            // Take the first, second, third derivative of x_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * (2.0 - 2.0 * alpha * r_A_x * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * r_A_x * (-2.0 * alpha * r_A_y) *  // x_a^2 deriv e^{-a r^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * r_A_x * (-2.0 * alpha * r_A_z) *  // x_a^2 deriv e^{-a r^2} / dz
                    exponential;
            // Take the first, second, third derivative of y_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * r_A_y * (-2.0 * alpha * r_A_x) *   // deriv e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * (2.0 - 2.0 * alpha * r_A_y * r_A_y) *  // deriv y_A^2 e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * r_A_y * (-2.0 * alpha * r_A_z) *  // deriv e^{-a z^2} / dz
                    exponential;
            // Take the first, second, third derivative of z_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * r_A_z * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * r_A_z * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * (2.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
            // Take the first, second, third derivative of x_a y_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_y *   // deriv x_a e^{-x} /dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    r_A_x * (1.0 - 2.0 * alpha * r_A_y * r_A_y) *  // deriv y_a e^{-y} /dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    r_A_x * r_A_y * (-2.0 * alpha * r_A_z) *  // deriv e^{-z} /dz
                    exponential;
            // Take the first, second, third derivative of x_a z_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_z *  // deriv x_a e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    r_A_x * r_A_z * (-2.0 * alpha * r_A_y) *   // deriv e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    r_A_x * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *  // deriv z_a e^{-a z^2} / dz
                    exponential;
            // Take the first, second, third derivative of y_a z_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    r_A_y * r_A_z * (-2.0 * alpha * r_A_x) *   // deriv e^{-a r^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_y * r_A_y) * r_A_z *  // deriv y_a e^{-a r^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    r_A_y * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *  // deriv z_a e^{-a r^2} / dz
                    exponential;
          }
          else if (angmom == -2) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2'] which is m=[0, 1, -1, 2, -2]
            double norm_const = chemtools::normalization_primitive_pure_d(g_constant_basis[iconst + i_prim]);
            // (x, y, z) derivative of ((2 z_A^2 - x_A^2 - y_A^2) / 2.0) e^{-a r^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                norm_const *
                    coeff_prim *
                    r_A_x * (-1 - alpha * ( 2.0 * r_A_z * r_A_z - r_A_x * r_A_x - r_A_y * r_A_y)) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    r_A_y * (-1 - alpha * ( 2.0 * r_A_z * r_A_z - r_A_x * r_A_x - r_A_y * r_A_y)) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    r_A_z * (2.0 - alpha * ( 2.0 * r_A_z * r_A_z - r_A_x * r_A_x - r_A_y * r_A_y)) *
                    exponential;
            // Derivataive of sqrt(3) x_A * z_A e^{-a r^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_z *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                norm_const *
                  coeff_prim *
                  sqrt(3.0) * r_A_x * r_A_z * (-2.0 * alpha * r_A_y) *
                  exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_x * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
            // Derivative of sqrt(3) y_A * z_A e^{-a r^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                norm_const *
                  coeff_prim *
                  sqrt(3.0) * r_A_y * r_A_z * (-2.0 * alpha * r_A_x) *
                  exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * (1.0 - 2.0 * alpha * r_A_y * r_A_y) * r_A_z *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_y * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
            // Derivative of sqrt(3) (x_A^2 - y_A^2) / 2.0  e^{-a r^2}
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_x * (1.0 - alpha * (r_A_x * r_A_x - r_A_y * r_A_y)) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_y * (-1.0 - alpha * (r_A_x * r_A_x - r_A_y * r_A_y)) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * (r_A_x * r_A_x - r_A_y * r_A_y) * (-alpha * r_A_z) *
                    exponential;;
            // (x, y, z) derivative respectively of sqrt(3) * x_A * y_A e^{-a r^2}
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_y *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_x * (1.0 - 2.0 * alpha * r_A_y * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(3.0) * r_A_x * r_A_y * (-2.0 * alpha * r_A_z) *
                    exponential;;
          }
          else if (angmom == 3) {
            // The ordering is ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
            // Take the first, second, third derivative of x_a^3 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 0) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_x*r_A_x + 3) *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x)*r_A_y *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x)*r_A_z *   // deriv x**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y_a^3  e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_y * r_A_y * r_A_y) *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 0) *
                    coeff_prim *
                    r_A_y*r_A_y*(-2*alpha*r_A_y*r_A_y + 3) *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_y * r_A_y * r_A_y)*r_A_z *   // d y**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of z_A^3  e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 3) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_z * r_A_z * r_A_z) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 3) *
                    coeff_prim *
                    -2*alpha*r_A_y*(r_A_z * r_A_z * r_A_z) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 3) *
                    coeff_prim *
                    r_A_z*r_A_z*(-2*alpha*r_A_z*r_A_z + 3) *   // d z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a y_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 0) *
                    coeff_prim *
                    r_A_y*r_A_y*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(-alpha*r_A_y*r_A_y + 1) *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*r_A_y*r_A_z *   // d x*y**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a^2 y e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(-alpha*r_A_x*r_A_x + 1) *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 0) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_y*r_A_y + 1) *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_x*r_A_y*r_A_z *   // deriv x**2*y*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a^2 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(-alpha*r_A_x*r_A_x + 1) *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 1) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_x*r_A_y*r_A_z *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_z*r_A_z + 1) *   // d x**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a z_a^2  e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 2) *
                    coeff_prim *
                    r_A_z*r_A_z*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 2) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*r_A_z*r_A_z *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(-alpha*r_A_z*r_A_z + 1) *   // d x*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y_a z_a^2  e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 2) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*r_A_z*r_A_z *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 2) *
                    coeff_prim *
                    r_A_z*r_A_z*(-2*alpha*r_A_y*r_A_y + 1) *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 2) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(-alpha*r_A_z*r_A_z + 1) *   // d y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y_a^2 z_a  e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 1) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*r_A_y*r_A_z *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 1) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(-alpha*r_A_y*r_A_y + 1) *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 1) *
                    coeff_prim *
                    r_A_y*r_A_y*(-2*alpha*r_A_z*r_A_z + 1) *   // d y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a y_a z_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 1) *
                    coeff_prim *
                    r_A_y*r_A_z*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 1) *
                    coeff_prim *
                    r_A_x*r_A_z*(-2*alpha*r_A_y*r_A_y + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 1) *
                    coeff_prim *
                    r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z + 1) *   // d x*y*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
          }
          else if (angmom == -3) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
            // This was done using wolframalpha for the formulas
            double norm_const = chemtools::normalization_primitive_pure_f(g_constant_basis[iconst + i_prim]);
            // (x, y, z) derivative of ((2 z_A^2 - 3 x_A^2 - 3 y_A^2) / 2.0) e^{-a r^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                norm_const *
                    coeff_prim *
                    r_A_x*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    r_A_y*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        (2*alpha*r_A_z*r_A_z*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 6*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            // Derivataive of sqrt(1.5) (4 z_a^2 x_A   - x_A^3  - y_A^2 x_A ) / 2.0 e^{-a r^2} , m=1
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) *
                    (
                        (2*alpha*r_A_x*r_A_x*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) - 3*r_A_x*r_A_x - r_A_y*r_A_y + 4*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) * r_A_x*r_A_y*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) - 1) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) * r_A_x*r_A_z*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) + 4) *
                    exponential;
            // Derivative of  sqrt(1.5) (4 z_a^2 y_A   - x_A^2 y_A  - y_A^3 ) / 2.0 e^{-a r^2}  m = -1
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) * (
                     r_A_x*r_A_y*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) - 1)
                     ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) * (
                      (2*alpha*r_A_y*r_A_y*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) - r_A_x*r_A_x - 3*r_A_y*r_A_y + 4*r_A_z*r_A_z)/2
                     ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(1.5) *  (
                    r_A_y*r_A_z*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 4*r_A_z*r_A_z) + 4)
                    ) *
                    exponential;
            //  Derivative of  sqrt(15) (x_A^2  - y_A^2) z / 2.0 e^{-a r^2}  m = 2
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * r_A_x*r_A_z*(-alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * r_A_y*r_A_z*(-alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * (-2*alpha*r_A_z*r_A_z*(r_A_x*r_A_x - r_A_y*r_A_y) + r_A_x*r_A_x - r_A_y*r_A_y)/2 *
                    exponential;
            //  Derivative of  sqrt(15) x y z         m = -2
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * r_A_y*r_A_z*(-2*alpha*r_A_x*r_A_x + 1) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * r_A_x*r_A_z*(-2*alpha*r_A_y*r_A_y + 1) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(15.0) * r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z + 1) *
                    exponential;
            //  Derivative of  sqrt(2.5) * (x^2 - 3 y^2) x / 2         m = 3
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * (
                    (-2*alpha*r_A_x*r_A_x*(r_A_x*r_A_x - 3*r_A_y*r_A_y) + 3*r_A_x*r_A_x - 3*r_A_y*r_A_y)/2
                )  *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * r_A_x*r_A_y*(-alpha*(r_A_x*r_A_x - 3*r_A_y*r_A_y) - 3) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * -alpha*r_A_x*r_A_z*(r_A_x*r_A_x - 3*r_A_y*r_A_y) *
                    exponential;
            //  Derivative of  sqrt(2.5) (3x^2 - y^2) y / 2        m = -3
            d_deriv_contracs[knumb_points * (icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * r_A_x*r_A_y*(-alpha*(3*r_A_x*r_A_x - r_A_y*r_A_y) + 3) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * (
                    (-2*alpha*r_A_y*r_A_y*(3*r_A_x*r_A_x - r_A_y*r_A_y) + 3*r_A_x*r_A_x - 3*r_A_y*r_A_y)/2
                ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) * -alpha*r_A_y*r_A_z*(3*r_A_x*r_A_x - r_A_y*r_A_y) *
                    exponential;
          }
          else if (angmom == 4) {
            // The ordering is ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy', 'xxzz', 'xxyz',
            //                    'xxyy', 'xxxz', 'xxxy', 'xxxx']
            // Take the first, second, third derivative of z^4 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 4) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_z * r_A_z * r_A_z * r_A_z) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 4) *
                    coeff_prim *
                    -2*alpha*r_A_y*(r_A_z * r_A_z * r_A_z * r_A_z) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  0, 0, 4) *
                    coeff_prim *
                    2*(r_A_z * r_A_z * r_A_z)*(-alpha*r_A_z*r_A_z + 2) *   // d z**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y z^3 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 3) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*(r_A_z * r_A_z * r_A_z) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 3) *
                    coeff_prim *
                    (r_A_z * r_A_z * r_A_z)*(-2*alpha*r_A_y*r_A_y + 1) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  0, 1, 3) *
                    coeff_prim *
                    r_A_y*r_A_z*r_A_z*(-2*alpha*r_A_z*r_A_z + 3) *   // d y*z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y^2 z^2 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 2) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*r_A_y*r_A_z*r_A_z *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 2) *
                    coeff_prim *
                    2*r_A_y*r_A_z*r_A_z*(-alpha*r_A_y*r_A_y + 1) *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  0, 2, 2) *
                    coeff_prim *
                    2*r_A_y*r_A_y*r_A_z*(-alpha*r_A_z*r_A_z + 1) *   // d y**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y^3 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 1) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_y * r_A_y * r_A_y)*r_A_z *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 1) *
                    coeff_prim *
                    r_A_y*r_A_y*r_A_z*(-2*alpha*r_A_y*r_A_y + 3) *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  0, 3, 1) *
                    coeff_prim *
                    (r_A_y * r_A_y * r_A_y)*(-2*alpha*r_A_z*r_A_z + 1) *   // d y**3*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of y^4 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 4, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_y * r_A_y * r_A_y * r_A_y) *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 4, 0) *
                    coeff_prim *
                    2*(r_A_y * r_A_y * r_A_y)*(-alpha*r_A_y*r_A_y + 2) *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  0, 4, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_y * r_A_y * r_A_y * r_A_y)*r_A_z *   // d y**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x z^3 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 3) *
                    coeff_prim *
                    (r_A_z * r_A_z * r_A_z)*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 3) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_y*(r_A_z * r_A_z * r_A_z) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  1, 0, 3) *
                    coeff_prim *
                    r_A_x*r_A_z*r_A_z*(-2*alpha*r_A_z*r_A_z + 3) *   // d x*z**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x y z^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 2) *
                    coeff_prim *
                    r_A_y*r_A_z*r_A_z*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 2) *
                    coeff_prim *
                    r_A_x*r_A_z*r_A_z*(-2*alpha*r_A_y*r_A_y + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  1, 1, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_y*r_A_z*(-alpha*r_A_z*r_A_z + 1) *   // d x*y*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x y^2 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 1) *
                    coeff_prim *
                    r_A_y*r_A_y*r_A_z*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_y*r_A_z*(-alpha*r_A_y*r_A_y + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  1, 2, 1) *
                    coeff_prim *
                    r_A_x*r_A_y*r_A_y*(-2*alpha*r_A_z*r_A_z + 1) *   // d x*y**2*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x y^3 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 3, 0) *
                    coeff_prim *
                    (r_A_y * r_A_y * r_A_y)*(-2*alpha*r_A_x*r_A_x + 1) *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 3, 0) *
                    coeff_prim *
                    r_A_x*r_A_y*r_A_y*(-2*alpha*r_A_y*r_A_y + 3) *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  1, 3, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*(r_A_y * r_A_y * r_A_y)*r_A_z *   // d x*y**3*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x^2 z^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_z*r_A_z*(-alpha*r_A_x*r_A_x + 1) *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 2) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_x*r_A_y*r_A_z*r_A_z *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  2, 0, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_x*r_A_z*(-alpha*r_A_z*r_A_z + 1) *   // d x**2*z**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x^2 yz e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_y*r_A_z*(-alpha*r_A_x*r_A_x + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*r_A_z*(-2*alpha*r_A_y*r_A_y + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim],  2, 1, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z + 1) *   // d x**2*y*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x^2 y^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 2, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_y*r_A_y*(-alpha*r_A_x*r_A_x + 1) *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 2, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_x*r_A_y*(-alpha*r_A_y*r_A_y + 1) *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 2, 0) *
                    coeff_prim *
                    -2*alpha*r_A_x*r_A_x*r_A_y*r_A_y*r_A_z *   // d x**2*y**2*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x^3 z e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*r_A_z*(-2*alpha*r_A_x*r_A_x + 3) *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 1) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x)*r_A_y*r_A_z *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 1) *
                    coeff_prim *
                    (r_A_x * r_A_x * r_A_x)*(-2*alpha*r_A_z*r_A_z + 1) *   // d x**3*z*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x^3 y e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 1, 0) *
                    coeff_prim *
                    r_A_x*r_A_x*r_A_y*(-2*alpha*r_A_x*r_A_x + 3) *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 1, 0) *
                    coeff_prim *
                    (r_A_x * r_A_x * r_A_x)*(-2*alpha*r_A_y*r_A_y + 1) *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 1, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x)*r_A_y*r_A_z *   // d x**3*y*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
            // Take the first, second, third derivative of x_a^4 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 4, 0, 0) *
                    coeff_prim *
                    2*(r_A_x * r_A_x * r_A_x)*(-alpha*r_A_x*r_A_x + 2) *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 4, 0, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x * r_A_x)*r_A_y *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 4, 0, 0) *
                    coeff_prim *
                    -2*alpha*(r_A_x * r_A_x * r_A_x * r_A_x)*r_A_z *   // d x**4*exp(-a*(x**2 + y**2 + z**2)) / dz
                    exponential;
          }
          else if (angmom == -4) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
            // This was done using wolframalpha for the formulas
            double norm_const = chemtools::normalization_primitive_pure_g(g_constant_basis[iconst + i_prim]);
            // (x, y, z) derivative of e^{-a r^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_x*(-alpha*(35*(r_A_z * r_A_z * r_A_z * r_A_z) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) * (r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) + 6*r_A_x*r_A_x + 6*r_A_y*r_A_y -
                        24*r_A_z*r_A_z)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_y*(-alpha*(35*(r_A_z * r_A_z * r_A_z * r_A_z) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) * (r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) +
                        6*r_A_x*r_A_x + 6*r_A_y*r_A_y - 24*r_A_z*r_A_z)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_z*(-alpha*(35*(r_A_z * r_A_z * r_A_z * r_A_z) -
                        30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) * (r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) -
                        24*r_A_x*r_A_x - 24*r_A_y*r_A_y + 16*r_A_z*r_A_z)/4
                    ) *
                    exponential;
            // Derivataive of sqrt(2.5) (7z^2 - 3 (x^2 + y^2 + z^2)) xz / 2.0 , m=1
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_z*(2*alpha*r_A_x*r_A_x*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) -
                        9*r_A_x*r_A_x - 3*r_A_y*r_A_y + 4*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_x*r_A_y*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) - 3)
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_x*(2*alpha*r_A_z*r_A_z*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) -
                        3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 12*r_A_z*r_A_z)/2
                    )*
                    exponential;
            // Derivataive of sqrt(2.5) (7z^2 - 3 (x^2 + y^2 + z^2)) xz / 2.0 , m= -1
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_x*r_A_y*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) - 3)
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_z*(2*alpha*r_A_y*r_A_y*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) - 3*r_A_x*r_A_x - 9*r_A_y*r_A_y + 4*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(2.5) *
                    (
                        r_A_y*(2*alpha*r_A_z*r_A_z*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 4*r_A_z*r_A_z) -
                        3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 12*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            // Derivative  of sqrt(5)  (7 * z**2 + r2) * (x**2 - y**2) * exp / 4,  m= 2
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_x*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) -
                        2*r_A_x*r_A_x + 6*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_y*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) +
                        2*r_A_y*r_A_y - 6*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_z*(r_A_x*r_A_x - r_A_y*r_A_y)*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) + 6)/2
                    ) *
                    exponential;
            // Derivataive of sqrt(5)     (7 * z**2 - r2) * x * y * exp / 2,        m= -2
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_y*(2*alpha*r_A_x*r_A_x*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) -
                        3*r_A_x*r_A_x - r_A_y*r_A_y + 6*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_x*(2*alpha*r_A_y*r_A_y*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) -
                        r_A_x*r_A_x - 3*r_A_y*r_A_y + 6*r_A_z*r_A_z)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(5.0) *
                    (
                        r_A_x*r_A_y*r_A_z*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 6*r_A_z*r_A_z) + 6)
                    ) *
                    exponential;
            // Derivataive of sqrt(35 / 2)      (x**2 - 3 * y**2) * x * z * exp / 2,       m= 3
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_z*(-2*alpha*r_A_x*r_A_x*(r_A_x*r_A_x - 3*r_A_y*r_A_y) + 3*r_A_x*r_A_x - 3*r_A_y*r_A_y)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_x*r_A_y*r_A_z*(-alpha*(r_A_x*r_A_x - 3*r_A_y*r_A_y) - 3)
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_x*(r_A_x*r_A_x - 3*r_A_y*r_A_y)*(-2*alpha*r_A_z*r_A_z + 1)/2
                    ) *
                    exponential;
            // Derivataive of sqrt(35 / 2)    (3 * x**2 - y**2) * y * z * exp / 2,       m= -3
            d_deriv_contracs[knumb_points * (icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_x*r_A_y*r_A_z*(-alpha*(3*r_A_x*r_A_x - r_A_y*r_A_y) + 3)
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_z*(-2*alpha*r_A_y*r_A_y*(3*r_A_x*r_A_x - r_A_y*r_A_y) + 3*r_A_x*r_A_x - 3*r_A_y*r_A_y)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 6) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0 / 2.0) *
                    (
                        r_A_y*(3*r_A_x*r_A_x - r_A_y*r_A_y)*(-2*alpha*r_A_z*r_A_z + 1)/2
                    ) *
                    exponential;
            // Derivataive of sqrt(35)     (x**4 - 6 * x**2 * y**2 + y**4) * exp / 8,,       m= 4
            d_deriv_contracs[knumb_points * (icontractions + 7) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0) *
                    (
                        r_A_x*(-alpha*((r_A_x * r_A_x * r_A_x * r_A_x) - 6*r_A_x*r_A_x*r_A_y*r_A_y + (r_A_y * r_A_y * r_A_y * r_A_y)) + 2*r_A_x*r_A_x -
                        6*r_A_y*r_A_y)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 7) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0) *
                    (
                        r_A_y*(-alpha*((r_A_x * r_A_x * r_A_x * r_A_x) - 6*r_A_x*r_A_x*r_A_y*r_A_y + (r_A_y * r_A_y * r_A_y * r_A_y)) - 6*r_A_x*r_A_x +
                        2*r_A_y*r_A_y)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 7) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35) *
                    (
                        -alpha*r_A_z*((r_A_x * r_A_x * r_A_x * r_A_x) - 6*r_A_x*r_A_x*r_A_y*r_A_y + (r_A_y * r_A_y * r_A_y * r_A_y))/4
                    ) *
                    exponential;

            // Derivataive of sqrt(35)      (x**2 - y**2) * x * y * exp / 2 ,       m= -4
            d_deriv_contracs[knumb_points * (icontractions + 8) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35) *
                    (
                        r_A_y*(-2*alpha*r_A_x*r_A_x*(r_A_x*r_A_x - r_A_y*r_A_y) + 3*r_A_x*r_A_x - r_A_y*r_A_y)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 8) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35) *
                    (
                        r_A_x*(-2*alpha*r_A_y*r_A_y*(r_A_x*r_A_x - r_A_y*r_A_y) + r_A_x*r_A_x - 3*r_A_y*r_A_y)/2
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 8) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35) *
                    (
                        -alpha*r_A_x*r_A_y*r_A_z*(r_A_x*r_A_x - r_A_y*r_A_y)
                    ) *
                    exponential;
          }// End angmoms.
        } // End going over contractions of a single segmented shell.
        // Update index that goes over each contraction.
        if(angmom == 0){
          icontractions += 1;
        }
        else if (angmom == 1) {
          icontractions += 3;
        }
        else if (angmom == 2) {
          icontractions += 6;
        }
        else if (angmom == -2) {
          icontractions += 5;
        }
        else if (angmom == 3) {
          icontractions += 10;
        }
        else if (angmom == -3) {
          icontractions += 7;
        }
        else if (angmom == 4) {
          icontractions += 15;
        }
        else if (angmom == -4) {
          icontractions += 9;
        }
      } // End updating segmented shell.

      // Update index of constant memory, add the number of exponents then number of angular momentum terms then
      //        add the number of coefficients.
      iconst += numb_primitives + numb_segment_shells + numb_segment_shells * numb_primitives;
    } // End Contractions
  } // End If statement.
}




__host__ std::vector<double> chemtools::evaluate_contraction_derivatives(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
) {

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();
  printf("Number of basis-functions %d \n", knbasisfuncs);

  // The output of the contractions in column-major order with shape (3, M, N).
  std::vector<double> h_contractions(3 * knbasisfuncs * knumb_points);

  // Declare Function Pointers
  d_func_t h_deriv_contractions_func;
  cudaMemcpyFromSymbol(&h_deriv_contractions_func, p_evaluate_deriv_contractions, sizeof(d_func_t));

  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
  chemtools::cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_deriv_contractions;
  printf("Evaluate derivative \n");
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions, sizeof(double) * 3 * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  chemtools::evaluate_scalar_quantity(
      molecular_basis,
      false,
      false,
      h_deriv_contractions_func,
      d_deriv_contractions,
      d_points,
      knumb_points,
      knbasisfuncs,
      threadsPerBlock,
      grid
  );
  printf("Transfer \n");
  // Transfer from device memory to host memory
  chemtools::cuda_check_errors(cudaMemcpy(&h_contractions[0],
                                       d_deriv_contractions,
                                       sizeof(double) * 3 * knumb_points * knbasisfuncs, cudaMemcpyDeviceToHost));

  cudaFree(d_points);
  cudaFree(d_deriv_contractions);

  return h_contractions;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_gradient(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points, const bool return_row
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> gradient = chemtools::evaluate_electron_density_gradient_handle(
      handle, iodata, h_points, knumb_points, return_row
  );
  cublasDestroy(handle);
  return gradient;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_gradient_handle(
    // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int knumb_points, const bool return_row
) {

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  //chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();
  //printf("Number of basis-functions %d \n", knbasisfuncs);


  /**
   * Note that the maximum memory requirement is 5NM + M^2 + NM doubles that this uses.
   * So if you want to split based on points, then since gradient is stored in column-order it is bit tricky.
   * Solving for 11Gb we have (5NM + M^2 + N + M)8 bytes = 11Gb 1e9 (since 1e9 bytes = 1GB) Solve for N to get
   * N = (11 * 10^9 - M - M^2) / (5M + M).  This is the optimal number of points that it can do.
   */
  size_t t_numb_pts = knumb_points;
  size_t t_nbasis = knbasisfuncs;
  size_t t_highest_number_of_bytes = (
      sizeof(double) * (5 * t_numb_pts * t_nbasis + t_nbasis * t_nbasis + 3 *  t_numb_pts)
  );
  size_t free_mem = 0;  // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  //printf("Total Free Memory Avaiable in GPU is %zu \n", free_mem);
  // Calculate how much memory can fit inside the GPU memory.
  size_t t_numb_chunks = t_highest_number_of_bytes / (free_mem - 500000000);
  // Calculate how many points we can compute with free memory minus 0.5 gigabyte for safe measures:
  //    This is calculated by solving (5 * N M + M^2 + 3N) * 8 bytes = Free memory (in bytes)  for N to get:
  size_t t_numb_pts_of_each_chunk = (((free_mem - 500000000) / (sizeof(double)))  - t_nbasis * t_nbasis) /
      (5 * t_nbasis + 3);
  if (t_numb_pts_of_each_chunk == 0 and t_numb_chunks > 1.0) {
    // Haven't handle this case yet
    assert(0);
  }

  // Iterate through each chunk of the data set.
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  size_t i_iter = 0;
  // The output of the electron density in row-major with shape (N, 3).
  std::vector<double> h_grad_electron_density_row(3 * knumb_points);

  // Function pointers and copy from device to host so that it can be evaluted over any size of basis-set.
  d_func_t h_contractions_func;
  cudaMemcpyFromSymbol(&h_contractions_func, chemtools::p_evaluate_contractions, sizeof(d_func_t));
  d_func_t h_deriv_contractions_func;
  cudaMemcpyFromSymbol(&h_deriv_contractions_func, chemtools::p_evaluate_deriv_contractions, sizeof(d_func_t));

  while(index_to_copy < knumb_points) {
  // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
//    printf("Number of points %zu \n", number_pts_iter);
//    printf("Maximal number of points in x-axis to do each chunk %zu \n", number_pts_iter);

    // Transfer grid points to GPU, this is in column order with shape (N, 3)
    //  Becasue h_points is in column-order and we're slicing based on the number of points that can fit in memory.
    //  Need to slice each (x,y,z) coordinate seperately.
    double *d_points;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * number_pts_iter));
    for(int i_slice = 0; i_slice < 3; i_slice++) {
      chemtools::cuda_check_errors(cudaMemcpy(&d_points[i_slice * number_pts_iter],
                                           &h_points[i_slice * knumb_points + index_to_copy],
                                           sizeof(double) * number_pts_iter,
                                           cudaMemcpyHostToDevice));
    }


    // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
    double *d_deriv_contractions;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions,
                                         sizeof(double) * 3 * number_pts_iter * knbasisfuncs));
    dim3 threadsPerBlock(128);
    dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
//    chemtools::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
//        d_deriv_contractions, d_points, number_pts_iter, knbasisfuncs
//    );
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_deriv_contractions_func,
        d_deriv_contractions,
        d_points,
        number_pts_iter,
        knbasisfuncs,
        threadsPerBlock,
        grid
    );
    //chemtools::print_matrix<<<1, 1>>>(d_deriv_contractions, knbasisfuncs, knumb_points);
    cudaDeviceSynchronize();

    // Allocate memory and calculate the contractions (without derivatives)
    double *d_contractions;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_contractions, sizeof(double) * t_nbasis * number_pts_iter));
    dim3 threadsPerBlock2(128);
    dim3 grid2((number_pts_iter + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_contractions_func,
        d_contractions,
        d_points,
        number_pts_iter,
        t_nbasis,
        threadsPerBlock2,
        grid2
    );
    //chemtools::print_matrix<<<1, 1>>>(d_deriv_contractions, knbasisfuncs, knumb_points);
    cudaDeviceSynchronize();
//    chemtools::evaluate_contractions_from_constant_memory_on_any_grid<<<grid2, threadsPerBlock2>>>(
//        d_contractions, d_points, number_pts_iter, knbasisfuncs
//    );

    // Free up points memory in device/gpu memory.
    cudaFree(d_points);


    // Allocate memory to hold the matrix-multiplcation between d_one_rdm and each `i`th derivative (i_deriv, M, N)
    double *d_temp_rdm_derivs;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs, sizeof(double) * number_pts_iter * knbasisfuncs));
    // Allocate device memory for gradient of electron density in column-major order.
    double *d_gradient_electron_density;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_gradient_electron_density, sizeof(double) * 3 * number_pts_iter));
    // For each derivative, calculate the derivative of electron density seperately.
    for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
      // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
      double *d_ith_deriv = &d_deriv_contractions[i_deriv * number_pts_iter * knbasisfuncs];

      // Transfer one-rdm from host/cpu memory to device/gpu memory.
      double *d_one_rdm;
      chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
      chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                                  sizeof(double), iodata.GetMOOneRDM(),
                                                  iodata.GetOneRdmShape(), d_one_rdm, iodata.GetOneRdmShape()));

      // Matrix multiple one-rdm with the ith derivative of contractions
      double alpha = 1.0;
      double beta = 0.0;
      chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              number_pts_iter, knbasisfuncs, knbasisfuncs,
                                              &alpha, d_ith_deriv, number_pts_iter,
                                              d_one_rdm, knbasisfuncs, &beta,
                                              d_temp_rdm_derivs, number_pts_iter));
      cudaFree(d_one_rdm);

      // Do a hadamard product with the original contractions.
      dim3 threadsPerBlock2(320);
      dim3 grid2((number_pts_iter * knbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
      chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(
          d_temp_rdm_derivs, d_contractions, knbasisfuncs, number_pts_iter
      );

      // Take the sum to get the ith derivative of the electron density. This is done via matrix-vector multiplcaiton
      // of ones
      thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
      double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
      chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, number_pts_iter, knbasisfuncs,
                                              &alpha, d_temp_rdm_derivs, number_pts_iter, deviceVecPtr, 1, &beta,
                                              &d_gradient_electron_density[i_deriv * number_pts_iter], 1));

      // Free up memory in this iteration for the next calculation of the derivative.
      all_ones.clear();
      all_ones.shrink_to_fit();
    }

    cudaFree(d_temp_rdm_derivs);
    cudaFree(d_deriv_contractions);
    cudaFree(d_contractions);

    // Multiply the derivative by two since electron density = sum | mo-contractions |^2
    dim3 threadsPerBlock3(320);
    dim3 grid3((3 * number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    chemtools::multiply_scalar<<< grid3, threadsPerBlock3>>>(d_gradient_electron_density, 2.0, 3 * number_pts_iter);

    // Transfer from column-major to row-major order
    if(return_row){
      double *d_gradient_clone;
      chemtools::cuda_check_errors(cudaMalloc((double **) &d_gradient_clone, sizeof(double) * 3 * number_pts_iter));
      chemtools::cuda_check_errors(
          cudaMemcpy(d_gradient_clone, d_gradient_electron_density,
                     sizeof(double) * 3 * number_pts_iter, cudaMemcpyDeviceToDevice)
      );
      const double alpha = 1.0;
      const double beta = 0.0;
      chemtools::cublas_check_errors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                              3, number_pts_iter,
                                              &alpha, d_gradient_electron_density, number_pts_iter,
                                              &beta, d_gradient_electron_density, 3,
                                              d_gradient_clone, 3));
      cudaFree(d_gradient_electron_density);

      // Transfer the gradient  of device memory to host memory in row-major order.
      chemtools::cuda_check_errors(cudaMemcpy(&h_grad_electron_density_row[3 * index_to_copy],
                                           d_gradient_clone,
                                           sizeof(double) * 3 * number_pts_iter, cudaMemcpyDeviceToHost));
      cudaFree(d_gradient_clone);
    }
    else {
      // Transfer the x-coordinate gradient of device memory to host memory in row-major order.
      chemtools::cuda_check_errors(cudaMemcpy(&h_grad_electron_density_row[index_to_copy],
                                           d_gradient_electron_density,
                                           sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
      // Transfer the y-coordinate
      chemtools::cuda_check_errors(cudaMemcpy(&h_grad_electron_density_row[t_numb_pts + index_to_copy],
                                           &d_gradient_electron_density[number_pts_iter],
                                           sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
      // Transfer the z-coordinate
      chemtools::cuda_check_errors(cudaMemcpy(&h_grad_electron_density_row[2 * t_numb_pts + index_to_copy],
                                           &d_gradient_electron_density[2 * number_pts_iter],
                                           sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
      cudaFree(d_gradient_electron_density);
    }

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.
  } // end while loop

  return h_grad_electron_density_row;
}
