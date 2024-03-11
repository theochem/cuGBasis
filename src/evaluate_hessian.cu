#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "../include/evaluate_gradient.cuh"
#include "../include/evaluate_hessian.cuh"
#include "../include/evaluate_density.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/cuda_basis_utils.cuh"
#include "../include/basis_to_gpu.cuh"

__device__ extern d_func_t chemtools::p_evaluate_sec_contractions = chemtools::evaluate_sec_deriv_contractions_from_constant_memory;

__device__ void chemtools::evaluate_sec_deriv_contractions_from_constant_memory(
    double* d_sec_deriv_contracs, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_contr_start
) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;  // map thread index to the index of hte points
  if (global_index < knumb_points) {
    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];

    // Setup the initial variables.
    int iconst = 0;                              // Index to go over constant memory.
    unsigned int icontractions = i_contr_start;  // Index to go over basis-set dimension in `d_sec_deriv_contracs`
    unsigned int numb_contracted_shells = (int) g_constant_basis[iconst++];
    for (int icontr_shell = 0; icontr_shell < numb_contracted_shells; icontr_shell++) {
      // Distance from the grid point to the center of the contracted shell.
      double r_A_x = (grid_x - g_constant_basis[iconst++]);
      double r_A_y = (grid_y - g_constant_basis[iconst++]);
      double r_A_z = (grid_z - g_constant_basis[iconst++]);
//      double radius_sq = pow(r_A_x, 2.0) + pow(r_A_y, 2.0) + pow(r_A_z, 2.0);

      int numb_segment_shells = (int) g_constant_basis[iconst++];
      int numb_primitives = (int) g_constant_basis[iconst++];
      // iconst from here=H+0 to H+(numb_primitives - 1) is the exponents, need this to constantly reiterate them.
      for (int i_segm_shell = 0; i_segm_shell < numb_segment_shells; i_segm_shell++) {
        // Add the number of exponents, then add extra coefficients to enumerate.
        int angmom = (int) g_constant_basis[iconst + numb_primitives + (numb_primitives + 1) * i_segm_shell];
        for (int i_prim = 0; i_prim < numb_primitives; i_prim++) {
          double coeff_prim = g_constant_basis[iconst + numb_primitives * (i_segm_shell + 1) + i_prim + 1 + i_segm_shell];
          double alpha = g_constant_basis[iconst + i_prim];
          double exponential = exp(-alpha * (r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z));

          if (angmom == 0) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    2*alpha*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    2*alpha*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_s(alpha) *
                    coeff_prim *
                    2*alpha*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
          }
          if (angmom == 1) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_p(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
          }
          if (angmom == 2) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    2*(2*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 3*alpha*r_A_x*r_A_x + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_z*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    2*(2*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 3*alpha*r_A_y*r_A_y + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    4*alpha*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_z*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    4*alpha*r_A_y*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
                    coeff_prim *
                    2*(2*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 3*alpha*r_A_z*r_A_z + 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    (-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    (-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    (-2*alpha*r_A_y*r_A_y + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
          }
          if (angmom == -2) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of (-x**2 - y**2 + 2*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-2*alpha*r_A_x*r_A_x*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1) + 2*alpha*r_A_x*r_A_x + alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) + 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-2*alpha*r_A_y*r_A_y*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1) + 2*alpha*r_A_y*r_A_y + alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(-alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-2*alpha*r_A_z*r_A_z*(alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) + 2) - 4*alpha*r_A_z*r_A_z + alpha*(r_A_x*r_A_x + r_A_y*r_A_y - 2*r_A_z*r_A_z) + 2)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*x*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_z*(6.92820323027551*alpha*r_A_x*r_A_x - 10.3923048454133)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(3.46410161513775*alpha*r_A_x*r_A_x - 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-3.46410161513775*alpha*r_A_x*r_A_x + 2*alpha*r_A_z*r_A_z*(3.46410161513775*alpha*r_A_x*r_A_x - 1.73205080756888) + 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_z*(6.92820323027551*alpha*r_A_y*r_A_y - 3.46410161513775)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(6.92820323027551*alpha*r_A_z*r_A_z - 3.46410161513775)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_z*(6.92820323027551*alpha*r_A_z*r_A_z - 10.3923048454133)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*y*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_y*r_A_z*(6.92820323027551*alpha*r_A_x*r_A_x - 3.46410161513775)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_z*(6.92820323027551*alpha*r_A_y*r_A_y - 3.46410161513775)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(6.92820323027551*alpha*r_A_z*r_A_z - 3.46410161513775)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_y*r_A_z*(6.92820323027551*alpha*r_A_y*r_A_y - 10.3923048454133)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-3.46410161513775*alpha*r_A_y*r_A_y + 2*alpha*r_A_z*r_A_z*(3.46410161513775*alpha*r_A_y*r_A_y - 1.73205080756888) + 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_y*r_A_z*(6.92820323027551*alpha*r_A_z*r_A_z - 10.3923048454133)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of (1.73205080756888*x**2 - 1.73205080756888*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (3.46410161513775*alpha*r_A_x*r_A_x*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1) - 3.46410161513775*alpha*r_A_x*r_A_x - 1.73205080756888*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    3.46410161513775*alpha*alpha*r_A_x*r_A_y*(r_A_x*r_A_x - r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    3.46410161513775*alpha*r_A_x*r_A_z*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (3.46410161513775*alpha*r_A_y*r_A_y*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1) + 3.46410161513775*alpha*r_A_y*r_A_y - 1.73205080756888*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    3.46410161513775*alpha*r_A_y*r_A_z*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(3.46410161513775*alpha*r_A_z*r_A_z - 1.73205080756888)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of 1.73205080756888*x*y*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(6.92820323027551*alpha*r_A_x*r_A_x - 10.3923048454133)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    (-3.46410161513775*alpha*r_A_x*r_A_x + 2*alpha*r_A_y*r_A_y*(3.46410161513775*alpha*r_A_x*r_A_x - 1.73205080756888) + 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(3.46410161513775*alpha*r_A_x*r_A_x - 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(6.92820323027551*alpha*r_A_y*r_A_y - 10.3923048454133)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(3.46410161513775*alpha*r_A_y*r_A_y - 1.73205080756888)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_d(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(6.92820323027551*alpha*r_A_z*r_A_z - 3.46410161513775)  *
                    exponential;
          }
          if (angmom == 3) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    2*r_A_x*(alpha*r_A_x*r_A_x*(2*alpha*r_A_x*r_A_x - 3) - 4*alpha*r_A_x*r_A_x + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_x*r_A_x*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_y*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    2*r_A_y*(alpha*r_A_y*r_A_y*(2*alpha*r_A_y*r_A_y - 3) - 4*alpha*r_A_y*r_A_y + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*r_A_z*r_A_z*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
                    coeff_prim *
                    2*r_A_z*(alpha*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3) - 4*alpha*r_A_z*r_A_z + 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    2*r_A_y*(2*alpha*r_A_x*r_A_x - 1)*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    2*r_A_x*(2*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 3*alpha*r_A_y*r_A_y + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    2*r_A_y*(2*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 3*alpha*r_A_x*r_A_x + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    2*r_A_x*(alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    2*r_A_z*(2*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 3*alpha*r_A_x*r_A_x + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    2*r_A_x*(alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    2*r_A_z*(2*alpha*r_A_x*r_A_x - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
                    coeff_prim *
                    2*r_A_x*(2*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 3*alpha*r_A_z*r_A_z + 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    2*r_A_z*(2*alpha*r_A_y*r_A_y - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
                    coeff_prim *
                    2*r_A_y*(2*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 3*alpha*r_A_z*r_A_z + 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    2*r_A_z*(2*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 3*alpha*r_A_y*r_A_y + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    2*r_A_y*(alpha*r_A_y*r_A_y - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    r_A_z*(2*alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    r_A_y*(2*alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    r_A_x*(2*alpha*r_A_y*r_A_y - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
          }
          if (angmom == -3) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*(-3*x**2 - 3*y**2 + 2*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(-2*alpha*r_A_x*r_A_x*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) + 6*alpha*r_A_x*r_A_x + alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(-alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) + 6)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(-2*alpha*r_A_z*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) - 4*alpha*r_A_z*r_A_z + alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(-2*alpha*r_A_y*r_A_y*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) + 6*alpha*r_A_y*r_A_y + alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(-2*alpha*r_A_z*r_A_z*(alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3) - 4*alpha*r_A_z*r_A_z + alpha*(3*r_A_x*r_A_x + 3*r_A_y*r_A_y - 2*r_A_z*r_A_z) - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(-6*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z - 6*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 4*alpha*alpha*pow(r_A_z, 4) + 9*alpha*r_A_x*r_A_x + 9*alpha*r_A_y*r_A_y - 14*alpha*r_A_z*r_A_z + 6)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*(-1.22474487139159*x**2 - 1.22474487139159*y**2 + 4.89897948556636*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    1.0*r_A_x*(-2.44948974278318*alpha*alpha*pow(r_A_x, 4) - 2.44948974278318*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y + 9.79795897113271*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z + 8.57321409974112*alpha*r_A_x*r_A_x + 3.67423461417477*alpha*r_A_y*r_A_y - 14.6969384566991*alpha*r_A_z*r_A_z - 3.67423461417477)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(2.44948974278318*alpha*r_A_x*r_A_x - 2*alpha*(alpha*r_A_x*r_A_x*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.83711730708738*r_A_x*r_A_x - 0.612372435695794*r_A_y*r_A_y + 2.44948974278318*r_A_z*r_A_z) - 1.22474487139159)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(-9.79795897113271*alpha*r_A_x*r_A_x - 2*alpha*(alpha*r_A_x*r_A_x*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.83711730708738*r_A_x*r_A_x - 0.612372435695794*r_A_y*r_A_y + 2.44948974278318*r_A_z*r_A_z) + 4.89897948556636)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(-2*alpha*r_A_y*r_A_y*(alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159) + 2.44948974278318*alpha*r_A_y*r_A_y + alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*r_A_z*(-2*alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 7.34846922834953)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(-2*alpha*r_A_z*r_A_z*(alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) + 4.89897948556636) - 9.79795897113271*alpha*r_A_z*r_A_z + alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) + 4.89897948556636)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*(-1.22474487139159*x**2 - 1.22474487139159*y**2 + 4.89897948556636*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(-2*alpha*r_A_x*r_A_x*(alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159) + 2.44948974278318*alpha*r_A_x*r_A_x + alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(-2*alpha*r_A_y*r_A_y*(alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159) + 2.44948974278318*alpha*r_A_y*r_A_y + alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 1.22474487139159)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*r_A_z*(-2*alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 7.34846922834953)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    1.0*r_A_y*(-2.44948974278318*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 2.44948974278318*alpha*alpha*pow(r_A_y, 4) + 9.79795897113271*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 3.67423461417477*alpha*r_A_x*r_A_x + 8.57321409974112*alpha*r_A_y*r_A_y - 14.6969384566991*alpha*r_A_z*r_A_z - 3.67423461417477)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(-9.79795897113271*alpha*r_A_y*r_A_y - 2*alpha*(alpha*r_A_y*r_A_y*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) - 0.612372435695794*r_A_x*r_A_x - 1.83711730708738*r_A_y*r_A_y + 2.44948974278318*r_A_z*r_A_z) + 4.89897948556636)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(-2*alpha*r_A_z*r_A_z*(alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) + 4.89897948556636) - 9.79795897113271*alpha*r_A_z*r_A_z + alpha*(1.22474487139159*r_A_x*r_A_x + 1.22474487139159*r_A_y*r_A_y - 4.89897948556636*r_A_z*r_A_z) + 4.89897948556636)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z*(3.87298334620742*x**2 - 3.87298334620742*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(7.74596669241483*alpha*r_A_x*r_A_x*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1) - 7.74596669241483*alpha*r_A_x*r_A_x - 3.87298334620742*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 3.87298334620742)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    7.74596669241483*alpha*alpha*r_A_x*r_A_y*r_A_z*(r_A_x*r_A_x - r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(7.74596669241483*alpha*r_A_z*r_A_z - 3.87298334620742)*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(7.74596669241483*alpha*r_A_y*r_A_y*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1) + 7.74596669241483*alpha*r_A_y*r_A_y - 3.87298334620742*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 3.87298334620742)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(7.74596669241483*alpha*r_A_z*r_A_z - 3.87298334620742)*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_z*(7.74596669241483*alpha*r_A_z*r_A_z*(r_A_x*r_A_x - r_A_y*r_A_y) - 11.6189500386223*r_A_x*r_A_x + 11.6189500386223*r_A_y*r_A_y)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of 3.87298334620742*x*y*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*r_A_z*(15.4919333848297*alpha*r_A_x*r_A_x - 23.2379000772445)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_z*(7.74596669241483*alpha*r_A_x*r_A_x - 3.87298334620742)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(7.74596669241483*alpha*r_A_x*r_A_x - 3.87298334620742)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*r_A_z*(15.4919333848297*alpha*r_A_y*r_A_y - 23.2379000772445)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(7.74596669241483*alpha*r_A_y*r_A_y - 3.87298334620742)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*r_A_z*(15.4919333848297*alpha*r_A_z*r_A_z - 23.2379000772445)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*(1.58113883008419*x**2 - 4.74341649025257*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    1.0*r_A_x*(3.16227766016838*alpha*alpha*pow(r_A_x, 4) - 9.48683298050514*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 11.0679718105893*alpha*r_A_x*r_A_x + 14.2302494707577*alpha*r_A_y*r_A_y + 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(9.48683298050514*alpha*r_A_x*r_A_x + 2*alpha*(alpha*r_A_x*r_A_x*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y) - 2.37170824512628*r_A_x*r_A_x + 2.37170824512628*r_A_y*r_A_y) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(alpha*r_A_x*r_A_x*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y) - 2.37170824512628*r_A_x*r_A_x + 2.37170824512628*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(2*alpha*r_A_y*r_A_y*(alpha*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y) + 4.74341649025257) + 9.48683298050514*alpha*r_A_y*r_A_y - alpha*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(alpha*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y) + 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_x*(1.58113883008419*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*(4.74341649025257*x**2 - 1.58113883008419*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_y*(2*alpha*r_A_x*r_A_x*(alpha*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) - 4.74341649025257) - 9.48683298050514*alpha*r_A_x*r_A_x - alpha*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) + 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    r_A_x*(2*alpha*r_A_y*r_A_y*(alpha*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) - 4.74341649025257) + 3.16227766016838*alpha*r_A_y*r_A_y - alpha*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) + 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*(alpha*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    1.0*r_A_y*(9.48683298050514*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 3.16227766016838*alpha*alpha*pow(r_A_y, 4) - 14.2302494707577*alpha*r_A_x*r_A_x + 11.0679718105893*alpha*r_A_y*r_A_y - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    2*alpha*r_A_z*(alpha*r_A_y*r_A_y*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y) - 2.37170824512628*r_A_x*r_A_x + 2.37170824512628*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_f(alpha) *
                    coeff_prim *
                    alpha*r_A_y*(4.74341649025257*r_A_x*r_A_x - 1.58113883008419*r_A_y*r_A_y)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
          }
          if (angmom == 4) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of z**4*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    2*alpha*pow(r_A_z, 4)*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*r_A_y*pow(r_A_z, 4)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_z*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    2*alpha*pow(r_A_z, 4)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*r_A_y*r_A_z*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    r_A_z*r_A_z*(4*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 2) - 10*alpha*r_A_z*r_A_z + 12)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    r_A_z*r_A_z*(-6*alpha*r_A_y*r_A_y + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 1, 3) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(alpha*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3) - 4*alpha*r_A_z*r_A_z + 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**2*z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    r_A_z*r_A_z*(4*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 6*alpha*r_A_y*r_A_y + 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    4*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 2, 2) *
                    coeff_prim *
                    r_A_y*r_A_y*(4*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 6*alpha*r_A_z*r_A_z + 2)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**3*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(alpha*r_A_y*r_A_y*(2*alpha*r_A_y*r_A_y - 3) - 4*alpha*r_A_y*r_A_y + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    r_A_y*r_A_y*(-2*alpha*r_A_y*r_A_y + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 3) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 3, 1) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y**4*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    2*alpha*pow(r_A_y, 4)*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    4*alpha*alpha*r_A_x*pow(r_A_y, 4)*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    r_A_y*r_A_y*(4*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 2) - 10*alpha*r_A_y*r_A_y + 12)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    4*alpha*r_A_y*r_A_y*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 4, 0) *
                    coeff_prim *
                    2*alpha*pow(r_A_y, 4)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    r_A_z*r_A_z*(-6*alpha*r_A_x*r_A_x + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 0, 3) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(alpha*r_A_z*r_A_z*(2*alpha*r_A_z*r_A_z - 3) - 4*alpha*r_A_z*r_A_z + 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    r_A_z*r_A_z*(-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 1, 2) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 3*alpha*r_A_z*r_A_z + 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**2*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    r_A_y*r_A_y*(-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 3*alpha*r_A_y*r_A_y + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(alpha*r_A_y*r_A_y - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 2, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y**3*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    r_A_y*r_A_y*(-6*alpha*r_A_x*r_A_x + 2*alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 1) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(alpha*r_A_y*r_A_y*(2*alpha*r_A_y*r_A_y - 3) - 4*alpha*r_A_y*r_A_y + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 1, 3, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_y*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*z**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    r_A_z*r_A_z*(4*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 6*alpha*r_A_x*r_A_x + 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_z*r_A_z*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    4*r_A_x*r_A_z*(alpha*r_A_x*r_A_x - 1)*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 9) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 0, 2) *
                    coeff_prim *
                    r_A_x*r_A_x*(4*alpha*r_A_z*r_A_z*(alpha*r_A_z*r_A_z - 1) - 6*alpha*r_A_z*r_A_z + 2)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    2*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 3*alpha*r_A_x*r_A_x + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(alpha*r_A_x*r_A_x - 1)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_y*r_A_y + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y - 1) + 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 10) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 1, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**2*y**2*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    r_A_y*r_A_y*(4*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 1) - 6*alpha*r_A_x*r_A_x + 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    4*r_A_x*r_A_y*(alpha*r_A_x*r_A_x - 1)*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_y*r_A_y*r_A_z*(alpha*r_A_x*r_A_x - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    r_A_x*r_A_x*(4*alpha*r_A_y*r_A_y*(alpha*r_A_y*r_A_y - 1) - 6*alpha*r_A_y*r_A_y + 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 11) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 2, 2, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*z*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    2*r_A_x*r_A_z*(alpha*r_A_x*r_A_x*(2*alpha*r_A_x*r_A_x - 3) - 4*alpha*r_A_x*r_A_x + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x - 3) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 12) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 0, 1) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**3*y*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    2*r_A_x*r_A_y*(alpha*r_A_x*r_A_x*(2*alpha*r_A_x*r_A_x - 3) - 4*alpha*r_A_x*r_A_x + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    r_A_x*r_A_x*(-2*alpha*r_A_x*r_A_x + 2*alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x - 3) + 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_y*r_A_y - 3)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 13) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 3, 1, 0) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_x*r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x**4*exp(-a*(x**2 + y**2 + z**2))
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    r_A_x*r_A_x*(4*alpha*r_A_x*r_A_x*(alpha*r_A_x*r_A_x - 2) - 10*alpha*r_A_x*r_A_x + 12)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_x*r_A_x*r_A_y*(alpha*r_A_x*r_A_x - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*r_A_x*r_A_x*r_A_x*r_A_z*(alpha*r_A_x*r_A_x - 2)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    2*alpha*pow(r_A_x, 4)*(2*alpha*r_A_y*r_A_y - 1)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    4*alpha*alpha*pow(r_A_x, 4)*r_A_y*r_A_z  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 14) + global_index] +=
                chemtools::normalization_primitive_g(alpha, 0, 0, 4) *
                    coeff_prim *
                    2*alpha*pow(r_A_x, 4)*(2*alpha*r_A_z*r_A_z - 1)  *
                    exponential;
          }
          if (angmom == -4) {
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of (35*z**4 - 30*z**2*(x**2 + y**2 + z**2) + 3*(x**2 + y**2 + z**2)**2)*exp(-a*(x**2 + y**2 + z**2))/8
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (2*alpha*r_A_x*r_A_x*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 6*r_A_x*r_A_x - 6*r_A_y*r_A_y + 24*r_A_z*r_A_z) - alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) + 12*r_A_x*r_A_x*(alpha*(-r_A_x*r_A_x - r_A_y*r_A_y + 4*r_A_z*r_A_z) + 1) + 6*r_A_x*r_A_x + 6*r_A_y*r_A_y - 24*r_A_z*r_A_z)/4  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(6*alpha*(-r_A_x*r_A_x - r_A_y*r_A_y + 4*r_A_z*r_A_z) + alpha*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 6*r_A_x*r_A_x - 6*r_A_y*r_A_y + 24*r_A_z*r_A_z) + 6)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(-8*alpha*(-3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 2*r_A_z*r_A_z) + alpha*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 6*r_A_x*r_A_x - 6*r_A_y*r_A_y + 24*r_A_z*r_A_z) - 24)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (2*alpha*r_A_y*r_A_y*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 6*r_A_x*r_A_x - 6*r_A_y*r_A_y + 24*r_A_z*r_A_z) - alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) + 6*r_A_x*r_A_x + 12*r_A_y*r_A_y*(alpha*(-r_A_x*r_A_x - r_A_y*r_A_y + 4*r_A_z*r_A_z) + 1) + 6*r_A_y*r_A_y - 24*r_A_z*r_A_z)/4  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(-8*alpha*(-3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 2*r_A_z*r_A_z) + alpha*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 6*r_A_x*r_A_x - 6*r_A_y*r_A_y + 24*r_A_z*r_A_z) - 24)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 0) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (2*alpha*r_A_z*r_A_z*(alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) + 24*r_A_x*r_A_x + 24*r_A_y*r_A_y - 16*r_A_z*r_A_z) - alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) + 3*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z)) - 24*r_A_x*r_A_x - 24*r_A_y*r_A_y - 16*r_A_z*r_A_z*(alpha*(-3*r_A_x*r_A_x - 3*r_A_y*r_A_y + 2*r_A_z*r_A_z) - 2) + 16*r_A_z*r_A_z)/4  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*(-4.74341649025257*x**2 - 4.74341649025257*y**2 + 6.32455532033676*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_x*r_A_z*(-18.9736659610103*alpha*alpha*pow(r_A_x, 4) - 18.9736659610103*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y + 25.298221281347*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z + 66.407830863536*alpha*r_A_x*r_A_x + 28.4604989415154*alpha*r_A_y*r_A_y - 37.9473319220206*alpha*r_A_z*r_A_z - 28.4604989415154)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(18.9736659610103*alpha*r_A_x*r_A_x - 2*alpha*(2*alpha*r_A_x*r_A_x*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 14.2302494707577*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y + 6.32455532033676*r_A_z*r_A_z) - 9.48683298050514)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (alpha*r_A_x*r_A_x*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 14.2302494707577*r_A_x*r_A_x - 4.74341649025257*r_A_y*r_A_y + 6.32455532033676*r_A_z*r_A_z) - 7.11512473537885*r_A_x*r_A_x - 2.37170824512628*r_A_y*r_A_y + r_A_z*r_A_z*(-25.298221281347*alpha*r_A_x*r_A_x + 12.6491106406735)/2 + 3.16227766016838*r_A_z*r_A_z)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(-2*alpha*r_A_y*r_A_y*(alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257) + 9.48683298050514*alpha*r_A_y*r_A_y + alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z*(alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257) - 12.6491106406735*alpha*r_A_z*r_A_z + alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 1) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_x*r_A_z*(-18.9736659610103*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z - 18.9736659610103*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 25.298221281347*alpha*alpha*pow(r_A_z, 4) + 28.4604989415154*alpha*r_A_x*r_A_x + 28.4604989415154*alpha*r_A_y*r_A_y - 88.5437744847146*alpha*r_A_z*r_A_z + 37.9473319220206)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*(-4.74341649025257*x**2 - 4.74341649025257*y**2 + 6.32455532033676*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(-2*alpha*r_A_x*r_A_x*(alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257) + 9.48683298050514*alpha*r_A_x*r_A_x + alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(-2*alpha*r_A_y*r_A_y*(alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257) + 9.48683298050514*alpha*r_A_y*r_A_y + alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z*(alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257) - 12.6491106406735*alpha*r_A_z*r_A_z + alpha*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_y*r_A_z*(-18.9736659610103*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 18.9736659610103*alpha*alpha*pow(r_A_y, 4) + 25.298221281347*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 28.4604989415154*alpha*r_A_x*r_A_x + 66.407830863536*alpha*r_A_y*r_A_y - 37.9473319220206*alpha*r_A_z*r_A_z - 28.4604989415154)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (alpha*r_A_y*r_A_y*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y*(4.74341649025257*r_A_x*r_A_x + 4.74341649025257*r_A_y*r_A_y - 6.32455532033676*r_A_z*r_A_z) - 4.74341649025257*r_A_x*r_A_x - 14.2302494707577*r_A_y*r_A_y + 6.32455532033676*r_A_z*r_A_z) - 2.37170824512628*r_A_x*r_A_x - 7.11512473537885*r_A_y*r_A_y + r_A_z*r_A_z*(-25.298221281347*alpha*r_A_y*r_A_y + 12.6491106406735)/2 + 3.16227766016838*r_A_z*r_A_z)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 2) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_y*r_A_z*(-18.9736659610103*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z - 18.9736659610103*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 25.298221281347*alpha*alpha*pow(r_A_z, 4) + 28.4604989415154*alpha*r_A_x*r_A_x + 28.4604989415154*alpha*r_A_y*r_A_y - 88.5437744847146*alpha*r_A_z*r_A_z + 37.9473319220206)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of (x**2 - y**2)*(-2.23606797749979*x**2 - 2.23606797749979*y**2 + 13.4164078649987*z**2)*exp(-a*(x**2 + y**2 + z**2))/4
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (-2.23606797749979*alpha*alpha*pow(r_A_x, 6) + 13.4164078649987*alpha*alpha*pow(r_A_x, 4)*r_A_z*r_A_z + 2.23606797749979*alpha*alpha*r_A_x*r_A_x*pow(r_A_y, 4) - 13.4164078649987*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y*r_A_z*r_A_z + 10.0623058987491*alpha*pow(r_A_x, 4) - 33.5410196624969*alpha*r_A_x*r_A_x*r_A_z*r_A_z - 1.11803398874989*alpha*pow(r_A_y, 4) + 6.70820393249937*alpha*r_A_y*r_A_y*r_A_z*r_A_z - 6.70820393249937*r_A_x*r_A_x + 6.70820393249937*r_A_z*r_A_z)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(-2*alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 8.94427190999916*r_A_x*r_A_x - 8.94427190999916*r_A_y*r_A_y)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(-26.8328157299975*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 2*alpha*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - 4.47213595499958*r_A_x*r_A_x + 13.4164078649987*r_A_z*r_A_z) + 26.8328157299975)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (-2.23606797749979*alpha*alpha*pow(r_A_x, 4)*r_A_y*r_A_y + 13.4164078649987*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y*r_A_z*r_A_z + 2.23606797749979*alpha*alpha*pow(r_A_y, 6) - 13.4164078649987*alpha*alpha*pow(r_A_y, 4)*r_A_z*r_A_z + 1.11803398874989*alpha*pow(r_A_x, 4) - 6.70820393249937*alpha*r_A_x*r_A_x*r_A_z*r_A_z - 10.0623058987491*alpha*pow(r_A_y, 4) + 33.5410196624969*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 6.70820393249937*r_A_y*r_A_y - 6.70820393249937*r_A_z*r_A_z)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(-26.8328157299975*alpha*(r_A_x*r_A_x - r_A_y*r_A_y) - 2*alpha*(alpha*(r_A_x*r_A_x - r_A_y*r_A_y)*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 4.47213595499958*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - 26.8328157299975)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 3) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (r_A_x*r_A_x - r_A_y*r_A_y)*(-2*alpha*r_A_z*r_A_z*(alpha*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 13.4164078649987) - 26.8328157299975*alpha*r_A_z*r_A_z + alpha*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 13.4164078649987)/2  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*(-2.23606797749979*x**2 - 2.23606797749979*y**2 + 13.4164078649987*z**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_x*r_A_y*(-8.94427190999916*alpha*alpha*pow(r_A_x, 4) - 8.94427190999916*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y + 53.665631459995*alpha*alpha*r_A_x*r_A_x*r_A_z*r_A_z + 31.3049516849971*alpha*r_A_x*r_A_x + 13.4164078649987*alpha*r_A_y*r_A_y - 80.4984471899924*alpha*r_A_z*r_A_z - 13.4164078649987)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (alpha*r_A_x*r_A_x*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - alpha*r_A_y*r_A_y*(2*alpha*r_A_x*r_A_x*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - 6.70820393249937*r_A_x*r_A_x - 2.23606797749979*r_A_y*r_A_y + 13.4164078649987*r_A_z*r_A_z) - 3.35410196624968*r_A_x*r_A_x + r_A_y*r_A_y*(8.94427190999916*alpha*r_A_x*r_A_x - 4.47213595499958)/2 - 1.11803398874989*r_A_y*r_A_y + 6.70820393249937*r_A_z*r_A_z)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(-53.665631459995*alpha*r_A_x*r_A_x - 2*alpha*(2*alpha*r_A_x*r_A_x*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - 6.70820393249937*r_A_x*r_A_x - 2.23606797749979*r_A_y*r_A_y + 13.4164078649987*r_A_z*r_A_z) + 26.8328157299975)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_x*r_A_y*(-8.94427190999916*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 8.94427190999916*alpha*alpha*pow(r_A_y, 4) + 53.665631459995*alpha*alpha*r_A_y*r_A_y*r_A_z*r_A_z + 13.4164078649987*alpha*r_A_x*r_A_x + 31.3049516849971*alpha*r_A_y*r_A_y - 80.4984471899924*alpha*r_A_z*r_A_z - 13.4164078649987)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(-53.665631459995*alpha*r_A_y*r_A_y - 2*alpha*(2*alpha*r_A_y*r_A_y*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) - 2.23606797749979*r_A_x*r_A_x - 6.70820393249937*r_A_y*r_A_y + 13.4164078649987*r_A_z*r_A_z) + 26.8328157299975)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 4) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(-2*alpha*r_A_z*r_A_z*(alpha*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 13.4164078649987) - 26.8328157299975*alpha*r_A_z*r_A_z + alpha*(2.23606797749979*r_A_x*r_A_x + 2.23606797749979*r_A_y*r_A_y - 13.4164078649987*r_A_z*r_A_z) + 13.4164078649987)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*z*(4.18330013267038*x**2 - 12.5499003980111*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_x*r_A_z*(16.7332005306815*alpha*alpha*pow(r_A_x, 4) - 50.1996015920445*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 58.5662018573853*alpha*r_A_x*r_A_x + 75.2994023880668*alpha*r_A_y*r_A_y + 25.0998007960223)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(50.1996015920445*alpha*r_A_x*r_A_x + 2*alpha*(2*alpha*r_A_x*r_A_x*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) - 12.5499003980111*r_A_x*r_A_x + 12.5499003980111*r_A_y*r_A_y) - 25.0998007960223)/2  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (-alpha*r_A_x*r_A_x*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) + alpha*r_A_z*r_A_z*(2*alpha*r_A_x*r_A_x*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) - 12.5499003980111*r_A_x*r_A_x + 12.5499003980111*r_A_y*r_A_y) + 6.27495019900557*r_A_x*r_A_x - 6.27495019900557*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y*(alpha*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) + 12.5499003980111) + 25.0998007960223*alpha*r_A_y*r_A_y - alpha*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) - 12.5499003980111)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)*(alpha*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y) + 12.5499003980111)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 5) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_z*(4.18330013267038*r_A_x*r_A_x - 12.5499003980111*r_A_y*r_A_y)*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of y*z*(12.5499003980111*x**2 - 4.18330013267038*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_y*r_A_z*(2*alpha*r_A_x*r_A_x*(alpha*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) - 12.5499003980111) - 25.0998007960223*alpha*r_A_x*r_A_x - alpha*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) + 12.5499003980111)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_z*(2*alpha*r_A_y*r_A_y*(alpha*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) - 12.5499003980111) + 8.36660026534076*alpha*r_A_y*r_A_y - alpha*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) + 12.5499003980111)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(2*alpha*r_A_z*r_A_z - 1)*(alpha*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) - 12.5499003980111)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    0.5*r_A_y*r_A_z*(50.1996015920445*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 16.7332005306815*alpha*alpha*pow(r_A_y, 4) - 75.2994023880668*alpha*r_A_x*r_A_x + 58.5662018573853*alpha*r_A_y*r_A_y - 25.0998007960223)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (-alpha*r_A_y*r_A_y*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) + alpha*r_A_z*r_A_z*(2*alpha*r_A_y*r_A_y*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y) - 12.5499003980111*r_A_x*r_A_x + 12.5499003980111*r_A_y*r_A_y) + 6.27495019900557*r_A_x*r_A_x - 6.27495019900557*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 6) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    alpha*r_A_y*r_A_z*(12.5499003980111*r_A_x*r_A_x - 4.18330013267038*r_A_y*r_A_y)*(2*alpha*r_A_z*r_A_z - 3)  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of (5.91607978309962*x**4 - 35.4964786985977*x**2*y**2 + 5.91607978309962*y**4)*exp(-a*(x**2 + y**2 + z**2))/8
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (2.95803989154981*alpha*alpha*pow(r_A_x, 6) - 17.7482393492988*alpha*alpha*pow(r_A_x, 4)*r_A_y*r_A_y + 2.95803989154981*alpha*alpha*r_A_x*r_A_x*pow(r_A_y, 4) - 13.3111795119741*alpha*pow(r_A_x, 4) + 44.3705983732471*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 1.4790199457749*alpha*pow(r_A_y, 4) + 8.87411967464942*r_A_x*r_A_x - 8.87411967464942*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    r_A_x*r_A_y*(2*alpha*(8.87411967464942*r_A_x*r_A_x - 2.95803989154981*r_A_y*r_A_y) + 2*alpha*(2*alpha*(0.739509972887452*pow(r_A_x, 4) - 4.43705983732471*r_A_x*r_A_x*r_A_y*r_A_y + 0.739509972887452*pow(r_A_y, 4)) - 2.95803989154981*r_A_x*r_A_x + 8.87411967464942*r_A_y*r_A_y) - 17.7482393492988)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(2*alpha*(0.739509972887452*pow(r_A_x, 4) - 4.43705983732471*r_A_x*r_A_x*r_A_y*r_A_y + 0.739509972887452*pow(r_A_y, 4)) - 2.95803989154981*r_A_x*r_A_x + 8.87411967464942*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (2.95803989154981*alpha*alpha*pow(r_A_x, 4)*r_A_y*r_A_y - 17.7482393492988*alpha*alpha*r_A_x*r_A_x*pow(r_A_y, 4) + 2.95803989154981*alpha*alpha*pow(r_A_y, 6) - 1.4790199457749*alpha*pow(r_A_x, 4) + 44.3705983732471*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 13.3111795119741*alpha*pow(r_A_y, 4) - 8.87411967464942*r_A_x*r_A_x + 8.87411967464942*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(2*alpha*(0.739509972887452*pow(r_A_x, 4) - 4.43705983732471*r_A_x*r_A_x*r_A_y*r_A_y + 0.739509972887452*pow(r_A_y, 4)) + 8.87411967464942*r_A_x*r_A_x - 2.95803989154981*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 7) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    alpha*(alpha*r_A_z*r_A_z*(2.95803989154981*pow(r_A_x, 4) - 17.7482393492988*r_A_x*r_A_x*r_A_y*r_A_y + 2.95803989154981*pow(r_A_y, 4)) - 1.4790199457749*pow(r_A_x, 4) + 8.87411967464942*r_A_x*r_A_x*r_A_y*r_A_y - 1.4790199457749*pow(r_A_y, 4))  *
                    exponential;
            // Taking sec derv (xx, xy, xz, yy, yz, zz) of x*y*(5.91607978309962*x**2 - 5.91607978309962*y**2)*exp(-a*(x**2 + y**2 + z**2))/2
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*0 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    1.0*r_A_x*r_A_y*(11.8321595661992*alpha*alpha*pow(r_A_x, 4) - 11.8321595661992*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 41.4125584816973*alpha*r_A_x*r_A_x + 17.7482393492988*alpha*r_A_y*r_A_y + 17.7482393492988)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*1 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    (11.8321595661992*alpha*alpha*pow(r_A_x, 4)*r_A_y*r_A_y - 11.8321595661992*alpha*alpha*r_A_x*r_A_x*pow(r_A_y, 4) - 5.91607978309962*alpha*pow(r_A_x, 4) - 8.88178419700125e-16*alpha*r_A_x*r_A_x*r_A_y*r_A_y + 5.91607978309962*alpha*pow(r_A_y, 4) + 8.87411967464942*r_A_x*r_A_x - 8.87411967464942*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*2 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    2*alpha*r_A_y*r_A_z*(5.91607978309962*alpha*r_A_x*r_A_x*(r_A_x*r_A_x - r_A_y*r_A_y) - 8.87411967464942*r_A_x*r_A_x + 2.95803989154981*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*3 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    1.0*r_A_x*r_A_y*(11.8321595661992*alpha*alpha*r_A_x*r_A_x*r_A_y*r_A_y - 11.8321595661992*alpha*alpha*pow(r_A_y, 4) - 17.7482393492988*alpha*r_A_x*r_A_x + 41.4125584816973*alpha*r_A_y*r_A_y - 17.7482393492988)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*4 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    2*alpha*r_A_x*r_A_z*(5.91607978309962*alpha*r_A_y*r_A_y*(r_A_x*r_A_x - r_A_y*r_A_y) - 2.95803989154981*r_A_x*r_A_x + 8.87411967464942*r_A_y*r_A_y)  *
                    exponential;
            d_sec_deriv_contracs[knumb_points * (knumb_contractions*5 + icontractions + 8) + global_index] +=
                chemtools::normalization_primitive_pure_g(alpha) *
                    coeff_prim *
                    alpha*r_A_x*r_A_y*(r_A_x*r_A_x - r_A_y*r_A_y)*(11.8321595661992*alpha*r_A_z*r_A_z - 5.91607978309962)  *
                    exponential;
          }

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


__host__ std::vector<double> chemtools::evaluate_contraction_second_derivative(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
) {

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();

  // The output of the contractions in column-major order with shape (3, M, N).
  std::vector<double> h_contractions(6 * knbasisfuncs * knumb_points);

  // Function pointers and copy from device to host so that it can be evaluted over any size of basis-set.
  d_func_t h_sec_deriv_contractions_func;
  cudaMemcpyFromSymbol(&h_sec_deriv_contractions_func, chemtools::p_evaluate_sec_contractions, sizeof(d_func_t));


  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
  chemtools::cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_sec_deriv_contr;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_sec_deriv_contr, sizeof(double) * 6 * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
//  chemtools::evaluate_sec_deriv_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
//      d_sec_deriv_contr, d_points, knumb_points, knbasisfuncs
//  );
  chemtools::evaluate_scalar_quantity(
      molecular_basis,
      false,
      false,
      h_sec_deriv_contractions_func,
      d_sec_deriv_contr,
      d_points,
      knumb_points,
      knbasisfuncs,
      threadsPerBlock,
      grid
  );

  // Transfer from device memory to host memory
  chemtools::cuda_check_errors(cudaMemcpy(&h_contractions[0],
                                          d_sec_deriv_contr,
                                          sizeof(double) * 6 * knumb_points * knbasisfuncs, cudaMemcpyDeviceToHost));

  cudaFree(d_points);
  cudaFree(d_sec_deriv_contr);
  return h_contractions;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_hessian(
    chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> hessian = chemtools::evaluate_electron_density_hessian_handle(
      handle, iodata, h_points, knumb_points, return_row
  );
  cublasDestroy(handle);
  return hessian;
}


__host__ static void chemtools::evaluate_first_term(
    cublasHandle_t& handle,
    const chemtools::MolecularBasis& basis,
    double* d_hessian,
    const double* const d_points,
    const double* const h_one_rdm,
    const size_t& numb_pts_iter,
    const size_t& knbasisfuncs
    ) {

  // Get the function pointer to the derivatives.
  d_func_t h_deriv_contractions_func;
  cudaMemcpyFromSymbol(&h_deriv_contractions_func, chemtools::p_evaluate_deriv_contractions, sizeof(d_func_t));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double *d_deriv_contractions;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions,
                                          sizeof(double) * 3 * numb_pts_iter * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((numb_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
//  chemtools::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
//      d_deriv_contractions, d_points, numb_pts_iter, knbasisfuncs
//  );
  chemtools::evaluate_scalar_quantity(
      basis,
      false,
      false,
      h_deriv_contractions_func,
      d_deriv_contractions,
      d_points,
      numb_pts_iter,
      knbasisfuncs,
      threadsPerBlock,
      grid
  );

  // Allocate memory to hold the matrix-multiplcation between d_one_rdm and each `i`th derivative (i_deriv, M, N)
  double *d_temp_rdm_derivs;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs, sizeof(double) * numb_pts_iter * knbasisfuncs));
  // For each derivative, calculate the derivative of electron density seperately.
  int i_sec_derivs = 0;
#pragma unroll
  for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
    // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
    double *d_ith_deriv = &d_deriv_contractions[i_deriv * numb_pts_iter * knbasisfuncs];

    // Transfer one-rdm from host/cpu memory to device/gpu memory.
    double *d_one_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
    chemtools::cublas_check_errors(cublasSetMatrix(knbasisfuncs, knbasisfuncs,
                                                   sizeof(double), h_one_rdm, //iodata.GetMOOneRDM(),
                                                   knbasisfuncs, d_one_rdm, knbasisfuncs));

    // Matrix multiple one-rdm with the ith derivative of contractions
    double alpha = 1.0;
    double beta = 0.0;
    chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               numb_pts_iter, knbasisfuncs, knbasisfuncs,
                                               &alpha, d_ith_deriv, numb_pts_iter,
                                               d_one_rdm, knbasisfuncs, &beta,
                                               d_temp_rdm_derivs, numb_pts_iter));
    cudaFree(d_one_rdm);

#pragma unroll
    for(int j_deriv = i_deriv; j_deriv < 3; j_deriv++) {
      // Get the jth derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
      double *d_jth_deriv = &d_deriv_contractions[j_deriv * numb_pts_iter * knbasisfuncs];

      // Copy the d_temp_rdm_derivs
      double* d_temp_rdm_derivs_copy;
      chemtools::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs_copy, sizeof(double) * numb_pts_iter * knbasisfuncs));
      chemtools::cuda_check_errors(
            cudaMemcpy(
                d_temp_rdm_derivs_copy, d_temp_rdm_derivs, sizeof(double) * knbasisfuncs * numb_pts_iter,
                cudaMemcpyDeviceToDevice
            )
        );

      // Do a hadamard product with the original contractions.
      dim3 threadsPerBlock2(320);
      dim3 grid2((numb_pts_iter * knbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
      chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(
          d_temp_rdm_derivs_copy, d_jth_deriv, knbasisfuncs, numb_pts_iter
      );

      // Take the sum. This is done via matrix-vector multiplication of ones
      thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
      double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
      chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, numb_pts_iter, knbasisfuncs,
                                                 &alpha, d_temp_rdm_derivs_copy, numb_pts_iter, deviceVecPtr, 1, &beta,
                                                 &d_hessian[i_sec_derivs * numb_pts_iter], 1));
//      chemtools::print_first_ten_elements<<<1, 1>>>(&d_hessian[i_sec_derivs * numb_pts_iter]);
//      cudaDeviceSynchronize();

      // Free up memory in this iteration for the next calculation of the derivative.
      all_ones.clear();
      all_ones.shrink_to_fit();

      i_sec_derivs += 1;
      cudaFree(d_temp_rdm_derivs_copy);
    }
  }
  cudaFree(d_temp_rdm_derivs);
  cudaFree(d_deriv_contractions);
}

__host__ std::vector<double> chemtools::evaluate_electron_density_hessian_handle(
    // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row
) {

  // Get the function pointers to the correct GPU functions
  d_func_t h_contractions_func;
  cudaMemcpyFromSymbol(&h_contractions_func, chemtools::p_evaluate_contractions, sizeof(d_func_t));
  d_func_t h_sec_deriv_contractions_func;
  cudaMemcpyFromSymbol(&h_sec_deriv_contractions_func, chemtools::p_evaluate_sec_contractions, sizeof(d_func_t));

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  int knbasisfuncs = molecular_basis.numb_basis_functions();

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
  //    This is calculated by solving (12MN + 9N + N + M) * 8 bytes = Free memory (in bytes)  for N to get:
  size_t t_numb_pts_of_each_chunk = (((free_mem - 500000000) / (sizeof(double)))  - t_nbasis) /
      (12 * t_nbasis + 9);
  if (t_numb_pts_of_each_chunk == 0 and t_numb_chunks > 1.0) {
    // Haven't handle this case yet
    assert(0);
  }

  // Iterate through each chunk of the data set.
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  size_t i_iter = 0;
  // The output of the electron density in col-major with shape (N, 3, 3).
  std::vector<double> h_hess_electron_density_col(9 * knumb_points);

  while(index_to_copy < knumb_points) {
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
    // printf("Number of pts iter %zu \n", number_pts_iter);

    // Allocate device memory for hessian (6, N) (row-order) of electron density in column-major order.
    //    Stores the upper-triangle component in the order: xx, xy, xz, yy, yz, zz
    //
    double *d_hessian_electron_density;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_hessian_electron_density, sizeof(double) * 6 * number_pts_iter));
    cudaMemset(d_hessian_electron_density, 0.0, sizeof(double) * 6 * number_pts_iter);

    // Transfer grid points to GPU, this is in column order with shape (N, 3)
    //  Because h_points is in column-order and we're slicing based on the number of points that can fit in memory.
    //  Need to slice each (x,y,z) coordinate seperately.
    double *d_points;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * number_pts_iter));
    for(int i_slice = 0; i_slice < 3; i_slice++) {
      chemtools::cuda_check_errors(cudaMemcpy(&d_points[i_slice * number_pts_iter],
                                              &h_points[i_slice * knumb_points + index_to_copy],
                                              sizeof(double) * number_pts_iter,
                                              cudaMemcpyHostToDevice));
    }

    // Allocate memory and calculate the second derivatives of the contractions (6, M, N)
    //   For some reason if this is after evaluate_first_term, it is incorrectly calculated, or if I put
    //   evaluate_first_time after doing teh second term, the derivative of the contractions is incorrect.
    double *d_second_derivs_contrs;
    chemtools::cuda_check_errors(
        cudaMalloc((double **) &d_second_derivs_contrs, sizeof(double) * 6 * knbasisfuncs * number_pts_iter)
    );
    dim3 threadsPerBlock(128);
    dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
//    chemtools::evaluate_sec_deriv_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
//        d_second_derivs_contrs, d_points, number_pts_iter, knbasisfuncs
//    );
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_sec_deriv_contractions_func,
        d_second_derivs_contrs,
        d_points,
        number_pts_iter,
        knbasisfuncs,
        threadsPerBlock,
        grid
    );
    cudaDeviceSynchronize();


    // Allocate memory and calculate the contractions (M, N) (row-major order)
    double *d_contractions;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_contractions, sizeof(double) * t_nbasis * number_pts_iter));
    dim3 threadsPerBlock2(256);
    dim3 grid2((number_pts_iter + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
//    chemtools::evaluate_contractions_from_constant_memory_on_any_grid<<<grid2, threadsPerBlock2>>>(
//        d_contractions, d_points, number_pts_iter, knbasisfuncs
//    );
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_contractions_func,
        d_contractions,
        d_points,
        number_pts_iter,
        knbasisfuncs,
        threadsPerBlock2,
        grid2
    );
    cudaDeviceSynchronize();

    /**
     * Compute first term of the Hessian:
     *
     *  \sum_{kl} c_{kl} \frac{\partial \psi_k}{\partial x_n} \frac{\partial \psi_l}[\partial x_m}
     */
    chemtools::evaluate_first_term(
        handle, molecular_basis, d_hessian_electron_density, d_points, iodata.GetMOOneRDM(),
        number_pts_iter, knbasisfuncs
    );
    cudaDeviceSynchronize();

    /**
     * Compute second term of the Hessian:
     *
     *  \sum_{kl} c_{kl} \psi_k \frac{\partial^2 \psi_l}[\partial x_n \partial  x_m}
     */
    // Transfer one-rdm from host/cpu memory to device/gpu memory.
    double *d_one_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
    chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                                   sizeof(double), iodata.GetMOOneRDM(),
                                                   iodata.GetOneRdmShape(), d_one_rdm, iodata.GetOneRdmShape()));

    // Allocate memory to hold the matrix-multiplcation between d_one_rdm and contraction
    double *d_one_rdm_contrs;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm_contrs, sizeof(double) * number_pts_iter * knbasisfuncs));
    // Matrix multiple one-rdm with the contractions
    double alpha = 1.0;
    double beta = 0.0;
    chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               number_pts_iter, knbasisfuncs, knbasisfuncs,
                                               &alpha, d_contractions, number_pts_iter,
                                               d_one_rdm, knbasisfuncs, &beta,
                                               d_one_rdm_contrs, number_pts_iter));
    cudaDeviceSynchronize();

    // Free one-rdm and contractions cause it isn't needed anymore
    cudaFree(d_one_rdm);
    cudaFree(d_contractions);
    cudaFree(d_points);

    // Create intermediate copy array for d_one_rdm_contrs
    double* d_copy_one_rdm_contrs;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_copy_one_rdm_contrs, sizeof(double) * number_pts_iter * knbasisfuncs));


    int i_sec_deriv = 0;
    for(int i_deriv = 0; i_deriv < 3; i_deriv++) {
      for(int j_deriv = i_deriv; j_deriv < 3; j_deriv++) {
        // Get the second derivative corresponding to taking the `i_deriv`th and `j_deriv`th derivative.
        double* d_sec_deriv = &d_second_derivs_contrs[i_sec_deriv * number_pts_iter * knbasisfuncs];

        // Copy `d_one_rdm_contrs` into `d_copy_one_rdm_contrs`
        chemtools::cuda_check_errors(
            cudaMemcpy(
                d_copy_one_rdm_contrs, d_one_rdm_contrs, sizeof(double) * knbasisfuncs * number_pts_iter,
                cudaMemcpyDeviceToDevice
            )
        );

        // Do a hadamard product with the contractions
        dim3 threadsPerBlock3(320);
        dim3 grid3((number_pts_iter * knbasisfuncs + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
        chemtools::hadamard_product<<<grid3, threadsPerBlock3>>>(
            d_copy_one_rdm_contrs, d_sec_deriv, knbasisfuncs, number_pts_iter
        );

        // Create intermediate array to hold the sum of "\sum c_{kl} sec deriv with contractions".
        double* d_copy_sum;
        chemtools::cuda_check_errors(cudaMalloc((double **) &d_copy_sum, sizeof(double) * number_pts_iter));


        // Take the sum to get the ith derivative of the electron density.
        //    This is done via matrix-vector multiplication of ones
        thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
        double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
        chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, number_pts_iter, knbasisfuncs,
                                                   &alpha, d_copy_one_rdm_contrs, number_pts_iter,
                                                   deviceVecPtr, 1, &beta,
                                                   d_copy_sum, 1));

        // Sum to device Hessian (column-order)
        chemtools::sum_two_arrays_inplace<<<grid2, threadsPerBlock2>>>(
            &d_hessian_electron_density[i_sec_deriv * number_pts_iter], d_copy_sum, number_pts_iter
        );

        // Free up memory in this iteration for the next calculation of the derivative.
        all_ones.clear();
        all_ones.shrink_to_fit();
        cudaFree(d_copy_sum);

        i_sec_deriv += 1;  // Update to move to the next second derivative
      }
    }

    cudaFree(d_contractions);
    cudaFree(d_second_derivs_contrs);
    cudaFree(d_copy_one_rdm_contrs);
    cudaFree(d_one_rdm_contrs);


    // Multiply the derivative by two since electron density = sum | mo-contractions |^2
    dim3 threadsPerBlock3(320);
    dim3 grid3((6 * number_pts_iter + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
    chemtools::multiply_scalar<<< grid3, threadsPerBlock3>>>(d_hessian_electron_density, 2.0, 6 * number_pts_iter);


    // Return column implies iteration through points goes first then derivatives
    // Transfer the xx of device memory to host memory in row-major order.
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[index_to_copy],
                                            d_hessian_electron_density,
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the xy-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the xz-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[2 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[2 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the yx-coordinate from xy-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[3 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the yy-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[4 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[3 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the yz-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[5 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[4 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the zx-coordinate from xz-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[6 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[2 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the zy-coordinate from yz-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[7 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[4 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    // Transfer the zz-coordinate
    chemtools::cuda_check_errors(cudaMemcpy(&h_hess_electron_density_col[8 * t_numb_pts + index_to_copy],
                                            &d_hessian_electron_density[5 * number_pts_iter],
                                            sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));
    cudaFree(d_hessian_electron_density);

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.

  } // end while loop


  // Cheap slow (but easier to code) method to return the row-major order
  if (return_row) {
    // Currently h_hess_electron_density is in column -order (N, 3, 3) so points first, then xx, xy, xz, yx, etc
    // mapping col-order (numbpts, deirv x_i, deriv x_j) -> global index: (n, i, j) = n + 3iN + jN
    // mapping (n, i, j) = 9n + 3i + j
    std::vector<double> h_hess_electron_density_row(9 * knumb_points);
    for(size_t n = 0; n < knumb_points; n++) {
      #pragma unroll
      for(size_t i = 0; i < 3; i++) {
        #pragma unroll
        for(size_t j = 0; j < 3; j++) {
          h_hess_electron_density_row[9 * n + 3 * i + j] =
              h_hess_electron_density_col[n + 3 * i * knumb_points + j * knumb_points];
        }
      }
    }
    return h_hess_electron_density_row;
  }

  return h_hess_electron_density_col;
}