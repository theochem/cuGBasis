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


__global__ void gbasis::evaluate_derivatives_contractions_from_constant_memory(
    double* d_deriv_contracs, const double* const d_points, const int knumb_points, const int knumb_contractions
) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;  // map thread index to the index of hte points
  if (global_index < knumb_points){
    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];

    // Setup the initial variables.
    int iconst = 0;                                                          // Index to go over constant memory.
    unsigned int icontractions = 0;                                         // Index to go over rows of d_contractions_array
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
          double exponential = exp(- alpha * ( pow(r_A_x, 2.0) + pow(r_A_y, 2.0) + pow(r_A_z, 2.0)));

          // If S, P, D or F orbital/
          if(angmom == 0) {
            // Calculate the first derivative, then second, then third
            // d_contractions_array is stored as shape (3, M, N) where 3 is the three derivatives, M number of
            // contractions and N is the number of points, this conversion in row-major order is
            // N(M i_x + i_y) + i_z, where i_x=0,1,2,   i_y=0,...,M-1,   i_z=0,...,N-1.
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                gbasis::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_x) *  // d e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                gbasis::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_y) *  // d e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                gbasis::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (-2.0 * alpha * r_A_z) *  // d e^{-a z^2} / dz
                    exponential;
          }
          else if (angmom == 1) {
            // First, second and third derivative of x_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_x * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_x * (-2.0 * alpha * r_A_z) *
                    exponential;
            // First, second and third derivative of y_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_y * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + (icontractions + 1)) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_y * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + (icontractions + 1)) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_y * (-2.0 * alpha * r_A_z) *
                    exponential;
            // First, second and third derivative of z_A e^{-a r_A^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_z * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + (icontractions + 2)) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    r_A_z * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + (icontractions + 2)) + global_index] +=
                gbasis::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
          }
          else if (angmom == 2) {
            // The ordering is xx, xy, xz, yy, yz, zz
            // Take the first, second, third derivative of x_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * (2.0 - 2.0 * alpha * r_A_x * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * r_A_x * (-2.0 * alpha * r_A_y) *  // x_a^2 deriv e^{-a r^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                    coeff_prim *
                    r_A_x * r_A_x * (-2.0 * alpha * r_A_z) *  // x_a^2 deriv e^{-a r^2} / dz
                    exponential;

            // Take the first, second, third derivative of x_a y_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 1) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_y *   // deriv x_a e^{-x} /dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 1) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    r_A_x * (1.0 - 2.0 * alpha * r_A_y * r_A_y) *  // deriv y_a e^{-y} /dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 1) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                    coeff_prim *
                    r_A_x * r_A_y * (-2.0 * alpha * r_A_z) *  // deriv e^{-z} /dz
                    exponential;
            // Take the first, second, third derivative of x_a z_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 2) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_x * r_A_x) * r_A_z *  // deriv x_a e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 2) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    r_A_x * r_A_z * (-2.0 * alpha * r_A_y) *   // deriv e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 2) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                    coeff_prim *
                    r_A_x * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *  // deriv z_a e^{-a z^2} / dz
                    exponential;
            // Take the first, second, third derivative of y_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 3) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * r_A_y * (-2.0 * alpha * r_A_x) *   // deriv e^{-a x^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 3) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * (2.0 - 2.0 * alpha * r_A_y * r_A_y) *  // deriv y_A^2 e^{-a y^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 3) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                    coeff_prim *
                    r_A_y * r_A_y * (-2.0 * alpha * r_A_z) *  // deriv e^{-a z^2} / dz
                    exponential;
            // Take the first, second, third derivative of y_a z_a e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 4) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    r_A_y * r_A_z * (-2.0 * alpha * r_A_x) *   // deriv e^{-a r^2} / dx
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 4) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    (1.0 - 2.0 * alpha * r_A_y * r_A_y) * r_A_z *  // deriv y_a e^{-a r^2} / dy
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 4) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                    coeff_prim *
                    r_A_y * (1.0 - 2.0 * alpha * r_A_z * r_A_z) *  // deriv z_a e^{-a r^2} / dz
                    exponential;
            // Take the first, second, third derivative of z_a^2 e^{-a r_a^2}
            d_deriv_contracs[knumb_points * (icontractions + 5) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * r_A_z * (-2.0 * alpha * r_A_x) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 5) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * r_A_z * (-2.0 * alpha * r_A_y) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 5) + global_index] +=
                gbasis::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                    coeff_prim *
                    r_A_z * (2.0 - 2.0 * alpha * r_A_z * r_A_z) *
                    exponential;
          }
          else if (angmom == -2) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2'] which is m=[0, 1, -1, 2, -2]
            double norm_const = gbasis::normalization_primitive_pure_d(g_constant_basis[iconst + i_prim]);
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
            // TODO
            assert(0);
          }
          else if (angmom == -3) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
            // This was done using wolframalpha for the formulas
            double norm_const = gbasis::normalization_primitive_pure_f(g_constant_basis[iconst + i_prim]);
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
            assert(0);
          }
          else if (angmom == -4) {
            // Negatives are s denoting sine and c denoting cosine.
            // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
            // This was done using wolframalpha for the formulas
            double norm_const = gbasis::normalization_primitive_pure_g(g_constant_basis[iconst + i_prim]);
            // (x, y, z) derivative of e^{-a r^2}
            d_deriv_contracs[knumb_points * icontractions + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_x*(-alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*pow(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z, 2)) + 6*r_A_x*r_A_x + 6*r_A_y*r_A_y -
                        24*r_A_z*r_A_z)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_y*(-alpha*(35*pow(r_A_z, 4) - 30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*pow(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z,2)) +
                        6*r_A_x*r_A_x + 6*r_A_y*r_A_y - 24*r_A_z*r_A_z)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions) + global_index] +=
                norm_const *
                    coeff_prim *
                    (
                        r_A_z*(-alpha*(35*pow(r_A_z, 4) -
                        30*r_A_z*r_A_z*(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z) +
                        3*pow(r_A_x*r_A_x + r_A_y*r_A_y + r_A_z*r_A_z,2)) -
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
                        r_A_x*(-alpha*(pow(r_A_x, 4) - 6*r_A_x*r_A_x*r_A_y*r_A_y + pow(r_A_y, 4)) + 2*r_A_x*r_A_x -
                        6*r_A_y*r_A_y)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions + icontractions + 7) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35.0) *
                    (
                        r_A_y*(-alpha*(pow(r_A_x, 4) - 6*r_A_x*r_A_x*r_A_y*r_A_y + pow(r_A_y, 4)) - 6*r_A_x*r_A_x +
                        2*r_A_y*r_A_y)/4
                    ) *
                    exponential;
            d_deriv_contracs[knumb_points * (knumb_contractions * 2 + icontractions + 7) + global_index] +=
                norm_const *
                    coeff_prim *
                    sqrt(35) *
                    (
                        -alpha*r_A_z*(pow(r_A_x, 4) - 6*r_A_x*r_A_x*r_A_y*r_A_y + pow(r_A_y, 4))/4
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
          // TODO
          assert(0);
        }
        else if (angmom == -3) {
          icontractions += 7;
        }
        else if (angmom == 4) {
          // TODO
          assert(0);
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


__host__ std::vector<double> gbasis::evaluate_contraction_derivatives(
    gbasis::IOData& iodata, const double* h_points, const int knumb_points
) {
  cudaFuncSetCacheConfig(gbasis::evaluate_contractions_from_constant_memory_on_any_grid, cudaFuncCachePreferL1);

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  gbasis::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  gbasis::add_mol_basis_to_constant_memory_array(molecular_basis, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();
  printf("Number of basis-functions %d \n", knbasisfuncs);

  // The output of the contractions in column-major order with shape (3, M, N).
  std::vector<double> h_contractions(3 * knbasisfuncs * knumb_points);

  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
  gbasis::cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_deriv_contractions;
  printf("Evaluate derivative \n");
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions, sizeof(double) * 3 * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
      d_deriv_contractions, d_points, knumb_points, knbasisfuncs
  );
  printf("Transfer \n");
  // Transfer from device memory to host memory
  gbasis::cuda_check_errors(cudaMemcpy(&h_contractions[0],
                                       d_deriv_contractions,
                                       sizeof(double) * 3 * knumb_points * knbasisfuncs, cudaMemcpyDeviceToHost));

  cudaFree(d_points);
  cudaFree(d_deriv_contractions);

  return h_contractions;
}

__host__ std::vector<double> gbasis::evaluate_electron_density_gradient(
    gbasis::IOData& iodata, const double* h_points, const int knumb_points
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> gradient = gbasis::evaluate_electron_density_gradient_handle(
      handle, iodata, h_points, knumb_points
  );
  cublasDestroy(handle);
  return gradient;
}


__host__ std::vector<double> gbasis::evaluate_electron_density_gradient_handle(
    // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
    cublasHandle_t& handle, gbasis::IOData& iodata, const double* h_points, const int knumb_points
) {
  // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
  cudaFuncSetCacheConfig(gbasis::evaluate_contractions_from_constant_memory_on_any_grid, cudaFuncCachePreferL1);

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  gbasis::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  //gbasis::add_mol_basis_to_constant_memory_array(molecular_basis, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();
  //printf("Number of basis-functions %d \n", knbasisfuncs);

  // The output of the electron density in row-major and column-major order, respectively, with shape (N, 3).
  std::vector<double> h_grad_electron_density(3 * knumb_points);

  /**
   * Note that the maximum memory requirement is 5NM + M^2 + N + M doubles that this uses.
   * So if you want to split based on points, then since gradient is stored in column-order it is bit tricky.
   * Solving for 11Gb we have (5NM + M^2 + N + M)8 bytes = 11Gb 1e9 (since 1e9 bytes = 1GB) Solve for N to get
   * N = (11 * 10^9 - M - M^2) / (5M + M).  This is the optimal number of points that it can do.
   */

  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
  gbasis::cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_deriv_contractions;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions, sizeof(double) * 3 * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
      d_deriv_contractions, d_points, knumb_points, knbasisfuncs
  );
  //gbasis::print_first_ten_elements<<<1, 1>>>(d_deriv_contractions);
//  printf("Print the contractions \n");
  //cudaDeviceSynchronize();
  //gbasis::print_matrix<<<1, 1>>>(d_deriv_contractions, knbasisfuncs, knumb_points);
  //cudaDeviceSynchronize();

  // Allocate memory and calculate the contractions (without derivatives)
  double *d_contractions;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_contractions, sizeof(double) * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock2(320);
  dim3 grid2((knumb_points + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
  gbasis::evaluate_contractions_from_constant_memory_on_any_grid<<<grid2, threadsPerBlock2>>>(
      d_contractions, d_points, knumb_points
  );

  // Free up points memory in device/gpu memory.
  cudaFree(d_points);

  // Transfer one-rdm from host/cpu memory to device/gpu memory.
  double *d_one_rdm;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
  gbasis::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                              sizeof(double), iodata.GetMOOneRDM(),
                                              iodata.GetOneRdmShape(), d_one_rdm,iodata.GetOneRdmShape()));

  // Allocate memory to hold the matrix-multiplcation between d_one_rdm and each `i`th derivative (i_deriv, M, N)
  double *d_temp_rdm_derivs;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs, sizeof(double) * knumb_points * knbasisfuncs));
  // Allocate device memory for gradient of electron density in column-major order.
  double *d_gradient_electron_density;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_gradient_electron_density, sizeof(double) * 3 * knumb_points));
  // For each derivative, calculate the derivative of electron density seperately.
  for(int i_deriv = 0; i_deriv < 3; i_deriv++) {
    // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
    double* d_ith_deriv = &d_deriv_contractions[i_deriv * knumb_points * knbasisfuncs];
//    if (i_deriv == 0) {
//      printf("Get the ith derivative %d \n", i_deriv);
//      cudaDeviceSynchronize();
//      gbasis::print_all<<<1, 1>>>(d_ith_deriv, 4 * 5);
//      cudaDeviceSynchronize();
//      printf("\n");
//    }
    // Matrix multiple one-rdm with the ith derivative of contractions
    double alpha = 1.0;
    double beta = 0.0;
    gbasis::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            knumb_points, knbasisfuncs, knbasisfuncs,
                                            &alpha, d_ith_deriv, knumb_points,
                                            d_one_rdm, knbasisfuncs, &beta,
                                            d_temp_rdm_derivs, knumb_points));
//    if (i_deriv == 0) {
//      printf("Matrix multiply with one_rdm %d \n", i_deriv);
//      cudaDeviceSynchronize();
//      gbasis::print_all<<<1, 1>>>(d_temp_rdm_derivs, 4 * 5);
//      cudaDeviceSynchronize();
//      printf("\n");
//    }

    // Do a hadamard product with the original contractions.
    dim3 threadsPerBlock2(320);
    dim3 grid2((knumb_points * knbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    gbasis::hadamard_product<<<grid2, threadsPerBlock2>>>(
        d_temp_rdm_derivs, d_contractions, knbasisfuncs, knumb_points
    );
//    if (i_deriv == 0) {
//      printf("Do Hadamard product with one-rdm \n");
//      cudaDeviceSynchronize();
//      gbasis::print_all<<<1, 1>>>(&d_temp_rdm_derivs[0], 4 * 5);
//      printf("\n");
//    }
    // Take the sum to get the ith derivative of the electron density. This is done via matrix-vector multiplcaiton
    // of ones
    thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
    double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
    gbasis::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, knumb_points, knbasisfuncs,
                                            &alpha, d_temp_rdm_derivs, knumb_points, deviceVecPtr, 1, &beta,
                                            &d_gradient_electron_density[i_deriv * knumb_points], 1));
//    if(i_deriv == 0) {
//      printf("ith derivative %d \n", i_deriv);
//      cudaDeviceSynchronize();
//      gbasis::print_all<<<1, 1>>>(&d_gradient_electron_density[i_deriv * knumb_points], 4 * 5);
//      printf("\n");
//    }
//    if(i_deriv == 0) {
//      printf("Actual ith derivative %d \n", i_deriv);
//      cudaDeviceSynchronize();
//      gbasis::print_all<<<1, 1>>>(d_ith_deriv_electron_density, 4 * 5);
//      printf("\n");
//    }

    // Free up memory in this iteration for the next calculation of the derivative.
    all_ones.clear();
    all_ones.shrink_to_fit();
  }

  cudaFree(d_temp_rdm_derivs);
  cudaFree(d_one_rdm);
  cudaFree(d_deriv_contractions);
  cudaFree(d_contractions);

  // Multiply the derivative by two since electron density = sum | mo-contractions |^2
  dim3 threadsPerBlock3(320);
  dim3 grid3((3 * knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::multiply_scalar<<< grid3, threadsPerBlock3>>>(d_gradient_electron_density, 2.0, 3 * knumb_points);

  // Transfer from column-major to row-major order
  double *d_gradient_clone;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_gradient_clone, sizeof(double) * 3 * knumb_points));
  gbasis::cuda_check_errors(
      cudaMemcpy(d_gradient_clone, d_gradient_electron_density,
                 sizeof(double) * 3 * knumb_points, cudaMemcpyDeviceToDevice)
  );
  const double alpha = 1.0;
  const double beta = 0.0;
  gbasis::cublas_check_errors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          3, knumb_points,
                                          &alpha, d_gradient_electron_density, knumb_points,
                                          &beta, d_gradient_electron_density, 3,
                                          d_gradient_clone, 3));
  cudaFree(d_gradient_electron_density);

  // Transfer the gradient  of device memory to host memory in row-major order.
  gbasis::cuda_check_errors(cudaMemcpy(h_grad_electron_density.data(),
                                       d_gradient_clone,
                                       sizeof(double) * 3 * knumb_points, cudaMemcpyDeviceToHost));
  cudaFree(d_gradient_clone);

  // Free-One Rdm and d_deriv contractions and destroy the cublas handle
  return h_grad_electron_density;
}
