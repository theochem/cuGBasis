#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#include "cublas_v2.h"

#include "../include/evaluate_density.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/cuda_basis_utils.cuh"
#include "../include/basis_to_gpu.cuh"


__device__ d_func_t chemtools::p_evaluate_contractions = chemtools::evaluate_contractions_from_constant_memory_on_any_grid;

/**
 * Only evaluates Cartesian Gaussians and Pure-Solid harmonics.
 *
 * d_contractions_array is a 2D array with size (M, N), where M=number of basis functions and N=number of grid points.
 * Each thread is associated to a column and fills out the entire column by iterating through the rows.
 * This is done by going through the constant memory holding the basis set information.
 *
 * TODO: Utilize shared memory to reduce the local memory overhead,
 */
__device__ void chemtools::evaluate_contractions(
    double* d_contractions_array,
    const double& grid_x,
    const double& grid_y,
    const double& grid_z,
    const int& knumb_points,
    unsigned int& global_index,
    const int& i_contr_start
    ) {
  // Setup the initial variables.
  int iconst = 0;                                                          // Index to go over constant memory.
  unsigned int icontractions = i_contr_start;                              // Index to go over rows of d_contractions_array
  unsigned int numb_contracted_shells = (int) g_constant_basis[iconst++];

  #pragma unroll
  for(int icontr_shell = 0; icontr_shell < numb_contracted_shells; icontr_shell++) {
    double r_A_x = (grid_x - g_constant_basis[iconst++]);
    double r_A_y = (grid_y - g_constant_basis[iconst++]);
    double r_A_z = (grid_z - g_constant_basis[iconst++]);
    //double radius_sq = pow(r_A_x, 2.0) + pow(r_A_y, 2.0) + pow(r_A_z, 2.0);
    int numb_segment_shells = (int) g_constant_basis[iconst++];
    int numb_primitives = (int) g_constant_basis[iconst++];
    // iconst from here=H+0 to H+(numb_primitives - 1) is the exponents, need this to constantly reiterate them.
    for(int i_segm_shell=0; i_segm_shell < numb_segment_shells; i_segm_shell++) {
      // Add the number of exponents, then add extra coefficients to enumerate.
      int angmom = (int) g_constant_basis[iconst + numb_primitives + (numb_primitives + 1) * i_segm_shell];
      for(int i_prim=0; i_prim < numb_primitives; i_prim++) {
        double coeff_prim = g_constant_basis[iconst + numb_primitives * (i_segm_shell + 1) + i_prim + 1 + i_segm_shell];
        double exponential = exp(-g_constant_basis[iconst + i_prim] *
            ( r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z));
        // If S, P, D or F orbital/
        if(angmom == 0) {
          d_contractions_array[global_index + icontractions * knumb_points] +=
              chemtools::normalization_primitive_s(g_constant_basis[iconst + i_prim]) *
                  coeff_prim *
                  exponential;
        }
        else if (angmom == 1) {
          d_contractions_array[global_index + icontractions * knumb_points] +=
              chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                  coeff_prim *
                  r_A_x *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                  coeff_prim *
                  r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              chemtools::normalization_primitive_p(g_constant_basis[iconst + i_prim]) *
                  coeff_prim *
                  r_A_z *
                  exponential;
        }
        else if (angmom == 2) {
          // The ordering is ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']   Old ordering: xx, xy, xz, yy, yz, zz
          d_contractions_array[global_index + icontractions * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 2, 0, 0) *
                  coeff_prim *
                  r_A_x * r_A_x *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 2, 0) *
                  coeff_prim *
                  r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 0, 2) *
                  coeff_prim *
                  r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 1, 0) *
                  coeff_prim *
                  r_A_x * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 1, 0, 1) *
                  coeff_prim *
                  r_A_x * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 5) * knumb_points] +=
              chemtools::normalization_primitive_d(g_constant_basis[iconst + i_prim], 0, 1, 1) *
                  coeff_prim *
                  r_A_y * r_A_z *
                  exponential;
        }
        else if (angmom == -2) {
          // Negatives are s denoting sine and c denoting cosine.
          // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2']
          double norm_const = chemtools::normalization_primitive_pure_d(g_constant_basis[iconst + i_prim]);
          d_contractions_array[global_index + icontractions * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_d(0, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_d(1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_d(-1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_d(2, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_d(-2, r_A_x, r_A_y, r_A_z) *
                  exponential;
        }
        else if (angmom == 3) {
          // The ordering is ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
          d_contractions_array[global_index + icontractions * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 3, 0, 0) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_x *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 3, 0) *
                  coeff_prim *
                  r_A_y * r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 0, 3) *
                  coeff_prim *
                  r_A_z * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 2, 0) *
                  coeff_prim *
                  r_A_x * r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 1, 0) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 5) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 2, 0, 1) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 6) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 0, 2) *
                  coeff_prim *
                  r_A_x * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 7) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 1, 2) *
                  coeff_prim *
                  r_A_y * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 8) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 0, 2, 1) *
                  coeff_prim *
                  r_A_y * r_A_y * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 9) * knumb_points] +=
              chemtools::normalization_primitive_f(g_constant_basis[iconst + i_prim], 1, 1, 1) *
                  coeff_prim *
                  r_A_x * r_A_y * r_A_z *
                  exponential;
        }
        else if (angmom == -3) {
          // ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
          double norm_const = chemtools::normalization_primitive_pure_f(g_constant_basis[iconst + i_prim]);
          d_contractions_array[global_index + icontractions * knumb_points] +=
                norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(0, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(-1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(2, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(-2, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 5) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(3, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 6) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_f(-3, r_A_x, r_A_y, r_A_z) *
                  exponential;
        }
        else if (angmom == 4) {
          // The ordering is ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy', 'xxzz',
          //                                                                'xxyz', 'xxyy', 'xxxz', 'xxxy', 'xxxx']
          d_contractions_array[global_index + icontractions * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 0, 0, 4) *
                  coeff_prim *
                  r_A_z * r_A_z * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 0, 1, 3) *
                  coeff_prim *
                  r_A_y * r_A_z * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 0, 2, 2) *
                  coeff_prim *
                  r_A_y * r_A_y * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 0, 3, 1) *
                  coeff_prim *
                  r_A_y * r_A_y * r_A_y * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 0, 4, 0) *
                  coeff_prim *
                  r_A_y * r_A_y * r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 5) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 1, 0, 3) *
                  coeff_prim *
                  r_A_x * r_A_z * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 6) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 1, 1, 2) *
                  coeff_prim *
                  r_A_x * r_A_y * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 7) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 1, 2, 1) *
                  coeff_prim *
                  r_A_x * r_A_y * r_A_y * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 8) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 1, 3, 0) *
                  coeff_prim *
                  r_A_x * r_A_y * r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 9) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 2, 0, 2) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_z * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 10) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 2, 1, 1) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_y * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 11) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 2, 2, 0) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_y * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 12) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 3, 0, 1) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_x * r_A_z *
                  exponential;
          d_contractions_array[global_index + (icontractions + 13) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 3, 1, 0) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_x * r_A_y *
                  exponential;
          d_contractions_array[global_index + (icontractions + 14) * knumb_points] +=
              chemtools::normalization_primitive_g(g_constant_basis[iconst + i_prim], 4, 0, 0) *
                  coeff_prim *
                  r_A_x * r_A_x * r_A_x * r_A_x *
                  exponential;
        }
        else if (angmom == -4) {
          // ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
          double norm_const = chemtools::normalization_primitive_pure_g(g_constant_basis[iconst + i_prim]);
          d_contractions_array[global_index + icontractions * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(0, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 1) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 2) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(-1, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 3) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(2, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 4) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(-2, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 5) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(3, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 6) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(-3, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 7) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(4, r_A_x, r_A_y, r_A_z) *
                  exponential;
          d_contractions_array[global_index + (icontractions + 8) * knumb_points] +=
              norm_const *
                  coeff_prim *
                  chemtools::solid_harmonic_function_g(-4, r_A_x, r_A_y, r_A_z) *
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
}


__device__ void chemtools::evaluate_contractions_from_constant_memory_on_any_grid(
    double* d_contractions_array, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_const_start
) {
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index < knumb_points) {
    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];
    // Evaluate the contractions and store it in d_contractions_array
    chemtools::evaluate_contractions(
        d_contractions_array, grid_x, grid_y, grid_z, knumb_points, global_index, i_const_start
    );
  }
}



/**
 * Evaluates the contractions on the cubic grid.
 *
 * This onyl works for one-dimensional threads.  The thread index i (0 <= i < (Nx Ny Nz)) is converted to
 *  block index form (ix, iy, iz) where Nx is the number of points in x-dimension.
 *
 *  The output iterates through z-axis, then y-axis then x-axis.
 *
 * TODO: Utilize shared memory to reduce the local memory overhead,
 */
__global__ void chemtools::evaluate_contractions_from_constant_memory_on_cubic_grid(
    double* d_contractions_array, const double3 d_klower_bnd, const double3 d_axes_array_col1,
    const double3 d_axes_array_col2, const double3 d_axes_array_col3, const int3 d_knumb_points
    ) {
  int knumb_points = d_knumb_points.x * d_knumb_points.y * d_knumb_points.z;
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index < knumb_points){
    // Compute the grid points by first converting from 1D index into 3D coordinate index.
    // It goes Z-axis first, then Y axis then X-axis
    unsigned int z_index = global_index % d_knumb_points.z;
    unsigned int y_index = (global_index / d_knumb_points.z) % d_knumb_points.y;
    unsigned int x_index = (global_index / (d_knumb_points.z * d_knumb_points.y));
    // Compute grid point based on origin + (indices).dot(axes),  where axes defines the shape of the grid.
    double grid_x = d_klower_bnd.x +
        (x_index * d_axes_array_col1.x) + (y_index * d_axes_array_col1.y) + (z_index * d_axes_array_col1.z);
    double grid_y = d_klower_bnd.y +
        (x_index * d_axes_array_col2.x) + (y_index * d_axes_array_col2.y) + (z_index * d_axes_array_col2.z) ;
    double grid_z = d_klower_bnd.z +
        (x_index * d_axes_array_col3.x) + (y_index * d_axes_array_col3.y) + (z_index * d_axes_array_col3.z);
    // Evaluate the contractions and store it in d_contractions_array
    chemtools::evaluate_contractions(
        d_contractions_array, grid_x, grid_y, grid_z, knumb_points, global_index
    );
  } // End If statement.
}


__host__ std::vector<double> chemtools::evaluate_electron_density_on_cubic(
    chemtools::IOData& iodata, const double3 klower_bnd, const double* kaxes_col, const int3 knumb_points, const bool disp )
    {

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  if (disp) {
    printf("Start transfer to constant memory \n");
  }
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  //chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false, false);
  int nbasisfuncs = molecular_basis.numb_basis_functions();
  if (disp) {
    printf("DOne transfer to constant memory \n");
  }

  // Total number of points.
  unsigned int number_of_pts = knumb_points.x * knumb_points.y * knumb_points.z;

  // Convert axes in column-major order into three double.
  const double3 axes_col1 = {kaxes_col[0], kaxes_col[1], kaxes_col[2]};
  const double3 axes_col2 = {kaxes_col[3], kaxes_col[4], kaxes_col[5]};
  const double3 axes_col3 = {kaxes_col[6], kaxes_col[7], kaxes_col[8]};
  if (disp) {
    printf("Axes (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", axes_col1.x, axes_col1.y, axes_col1.z,
             axes_col2.x, axes_col2.y, axes_col2.z, axes_col3.x, axes_col3.y, axes_col3.z);
    printf("Total shape is (%d, %d, %d)\n", knumb_points.x, knumb_points.y, knumb_points.z);
  }

  // Calculate the number of bytes it would take memory of
  //   The highest memory operation is doing Y=XA, where X is contraction array, A is one-rdm
  //   You need to store two matrices of size (MxN) where M is number of contractions
  //   and N is the number of points and one matrix of size (MxM) for the one-rdm.
  //   Take 12 Gigabyte GPU, then convert it into bytes
  size_t t_numb_pts_x = knumb_points.x;
  size_t t_numb_pts_y = knumb_points.y;
  size_t t_numb_pts_z = knumb_points.z;
  size_t t_numb_pts = number_of_pts;
  size_t t_nbasis = nbasisfuncs;
  size_t t_highest_number_of_bytes = sizeof(double) * (2 * t_numb_pts * t_nbasis + t_nbasis * t_nbasis);
  size_t free_mem = 0;    // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  free_mem -= 500000000;  // Subtract 0.5 GB for safe measures
  // Calculate how much memory can fit inside a  GPU memory.
  size_t t_numb_chunks = t_highest_number_of_bytes / free_mem;
  // Calculate how many points we can compute in the x-axis (keeping y-axis and z-axis fixed) with free memory
  //    This is calculated by solving (2 * N_x N_y N_z M + M^2) * 8 bytes = 11.5 Gb(in bytes)  for N_x to get:
  //    The reasoning on the x-axis is because we are storing the cubic grid that iterates z-axis first, then y-axis
  size_t t_numb_pts_of_each_chunk_x = ((free_mem / (sizeof(double)))  - t_nbasis * t_nbasis) /
      (2 * t_numb_pts_y * t_numb_pts_z * t_nbasis);
  if (t_numb_pts_of_each_chunk_x == 0 and t_numb_chunks > 1.0) {
       // Haven't handle this case yet
  }

  // Electron density in global memory and create the handles for using cublas.
//  double* h_electron_density;
//  h_electron_density = (double *) malloc(sizeof(double) * number_of_pts);
  std::vector<double> h_electron_density(number_of_pts);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Iterate through each chunk of the data set.
  double3 lower_bnd_iter = {klower_bnd.x, klower_bnd.y, klower_bnd.z};
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  //for(size_t i_iter = 0; i_iter < t_numb_chunks + 1; i_iter++) {
  size_t i_iter = 0;
  while(index_to_copy < number_of_pts) {
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_points_iter_x = std::min(
        t_numb_pts_x - i_iter * t_numb_pts_of_each_chunk_x, t_numb_pts_of_each_chunk_x
        );
    size_t number_points_iter = number_points_iter_x * t_numb_pts_y * t_numb_pts_z;
    int3 numb_points_iter = {static_cast<int>(number_points_iter_x), knumb_points.y, knumb_points.z};
    size_t total_numb_points_iter_bytes = number_points_iter * sizeof(double);
    size_t total_contraction_arr_iter_bytes = total_numb_points_iter_bytes * t_nbasis;
    if (disp) {
      printf("Number of chunks %zu \n", t_numb_chunks);
      printf("Maximal number of points to do each chunk %zu \n", number_points_iter);
      printf("Maximal number of points in x-axis to do each chunk %zu \n", number_points_iter_x);
    }

    //printf("Iter %d and lower bound  (%f, %f, %f) \n", i_iter, lower_bnd_iter.x, lower_bnd_iter.y, lower_bnd_iter.z);
    //printf("Number of points %zu, %zu, %zu \n", number_points_iter_x, t_numb_pts_y, t_numb_pts_z);
    //printf("Maximal number of points to do each chunk %zu \n", number_points_iter);
    //printf("Maximal number of points in x-axis to do each chunk %zu \n", number_points_iter_x);

    // Allocate device memory for contractions array, and set all elements to zero via cudaMemset.
    //    The contraction array rows are the atomic orbitals and columns are grid points and is stored in row-major order.
    double *d_contractions;
    cudaError err = cudaMalloc((double **) &d_contractions, total_contraction_arr_iter_bytes);
    // If error (most likely allocation error)
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating contractions, actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }
    chemtools::cuda_check_errors(cudaMemset(d_contractions, 0, total_contraction_arr_iter_bytes));

    // Evaluate contractions. The number of threads is maximal and the number of thread blocks is calculated.
    int ilen = 128;  // 128 320 1024
    dim3 threadsPerBlock(ilen);
    dim3 grid((number_points_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    if (disp) {
      printf("Start contractions %d \n ", grid.x);
    }
    chemtools::evaluate_contractions_from_constant_memory_on_cubic_grid<<<grid, threadsPerBlock>>>(
        d_contractions, lower_bnd_iter, axes_col1,
            axes_col2, axes_col3, numb_points_iter
    );
    //cudaDeviceSynchronize();
    //chemtools::print_matrix<<<1, 1>>>(d_contractions, t_nbasis, number_points_iter);
    //cudaDeviceSynchronize();
    if (disp) {
      printf("Done contractions \n ");
    }

    // Allocate device memory for the one_rdm. Number of atomic orbitals is equal to number of molecular obitals
    double *d_one_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, nbasisfuncs * nbasisfuncs * sizeof(double)));
    chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(),
                                                iodata.GetOneRdmShape(),
                                                sizeof(double),
                                                iodata.GetMOOneRDM(),
                                                iodata.GetOneRdmShape(),
                                                d_one_rdm,
                                                iodata.GetOneRdmShape()));
    //debug_print_all_row_order<<<1, 1>>>(d_one_rdm, 2, iodata.GetOneRdmShape());
    //cudaDeviceSynchronize();
    if (disp) {
      printf("Finish allocating one_rdm \n");
    }

    // Allocate device memory for the array holding the matrix multiplication of one_rdm and contraction array.
    double *d_final;
    err = cudaMalloc((double **) &d_final, total_contraction_arr_iter_bytes);
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating d_final in evaluating density, "
                   "actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cudaFree(d_one_rdm);
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }

    // Matrix multiplcation of one rdm with the contractions array. Everything is in row major order.
    double alpha = 1.;
    double beta = 0.;
    chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            number_points_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                            &alpha, d_contractions, number_points_iter,
                                            d_one_rdm, iodata.GetOneRdmShape(), &beta,
                                            d_final, number_points_iter));
//    printf("Finish matrimx ultiplcaitn onre rdm with contractions \n");
    cudaFree(d_one_rdm); // one body reduced density matrix isn't needed anymore.
//    printf("Free one rdm \n");

    // Do a hadamard product between d_final and d_contraction and store it in d_final. Maximal
    dim3 threadsPerBlock2(320);
    dim3 grid2((number_points_iter * nbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(d_final, d_contractions,
                                                          nbasisfuncs, number_points_iter);
    cudaFree(d_contractions);  // d_contractions is no longer needed.

    // Allocate device memory for electron density.
    double *d_electron_density;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_electron_density, total_numb_points_iter_bytes));

    // Sum up the columns of d_final to get the electron density. This is done by doing a matrix multiplcation of
    //    all ones of the transpose of d_final. Here I'm using the fact that d_final is in row major order.
//    printf("Sum up the columns \n ");
    thrust::device_vector<double> all_ones(sizeof(double) * nbasisfuncs, 1.0);
    double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
    chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, number_points_iter, nbasisfuncs,
                                            &alpha, d_final, number_points_iter, deviceVecPtr, 1, &beta,
                                            d_electron_density, 1));
    cudaFree(d_final);
//    printf("Done transfer to host \n");

    // Transfer electron density from device memory to host memory.
    //    Since I'm computing a sub-grid at a time, need to update the index h_electron_density, accordingly.
    chemtools::cuda_check_errors(cudaMemcpy(&h_electron_density[0] + index_to_copy,
                                         d_electron_density,
                                         total_numb_points_iter_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_electron_density);
    //printf("Electron density points %.10f %.10f %.10f %.10f \n", h_electron_density[index_to_copy],
    //       h_electron_density[index_to_copy + 1], h_electron_density[index_to_copy + 2],
    //       h_electron_density[index_to_copy + 3]);
    //printf("Electron density points %.10f %.10f %.10f %.10f \n", h_electron_density[1 * 174 * 229 * 83  ],
    //       h_electron_density[1 * 174 * 229 * 83 + 1], h_electron_density[1 * 174 * 229 * 83 + 2],
    //       h_electron_density[1 * 174 * 229 * 83 + 3]);
    //printf("Electron density points before %.10f %.10f %.10f %.10f \n", h_electron_density[1 * 174 * 229 * 83  - 1],
    //       h_electron_density[1 * 174 * 229 * 83  - 2], h_electron_density[1 * 174 * 229 * 83  - 3],
    //       h_electron_density[1 * 174 * 229 * 83  - 4]);
    //printf("Electron density 2 %.13f%.10f \n", h_electron_density[234* 229 * 83], h_electron_density[234 * 229 * 83 - 1]);

    // Update lower-bound of the grid for the next iteration
    lower_bnd_iter.x += number_points_iter_x * axes_col1.x;
    lower_bnd_iter.y += number_points_iter_x * axes_col2.x;
    lower_bnd_iter.z += number_points_iter_x * axes_col3.x;
    index_to_copy += number_points_iter;
    i_iter += 1;  // Update the index for each iteration.
  }
  cublasDestroy(handle); // cublas handle is no longer needed infact most of
  return h_electron_density;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_on_any_grid(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points)
{
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> density = chemtools::evaluate_electron_density_on_any_grid_handle(
      handle, iodata, h_points, knumb_points
  );
  cublasDestroy(handle); // cublas handle is no longer needed infact most of
  return density;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_on_any_grid_handle(
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int knumb_points
) {

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  //chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false, false);
  int nbasisfuncs = molecular_basis.numb_basis_functions();

  // Calculate the number of bytes it would take memory of
  //   The highest memory operation is doing Y=XA, where X is contraction array, A is one-rdm
  //   You need to store two matrices of size (MxN) where M is number of contractions
  //   and N is the number of points and one matrix of size (MxM) for the one-rdm.
  //   Take 12 Gigabyte GPU, then convert it into bytes
  size_t t_numb_pts = knumb_points;
  size_t t_nbasis = nbasisfuncs;
  size_t t_highest_number_of_bytes = sizeof(double) * (2 * t_numb_pts * t_nbasis + t_nbasis * t_nbasis);
  size_t free_mem = 0;   // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  // Calculate how much memory can fit inside a GPU memory.
  size_t t_numb_chunks = t_highest_number_of_bytes / (free_mem - 500000000);
  // Calculate how many points we can compute with 11.5 Gb:
  //    This is calculated by solving (2 * N M + M^2) * 8 bytes = 11.5 Gb(in bytes)  for N to get:
  size_t t_numb_pts_of_each_chunk = (((free_mem - 500000000) / (sizeof(double)))  - t_nbasis * t_nbasis) / (2 * t_nbasis);
  if (t_numb_pts_of_each_chunk == 0 and t_numb_chunks > 1.0) {
    // Haven't handle this case yet
    assert(0);
  }

  // Electron density in global memory and create the handles for using cublas.
  std::vector<double> h_electron_density(knumb_points);

  // Function pointers and copy from device to host
  d_func_t h_contractions_func;
  cudaMemcpyFromSymbol(&h_contractions_func, p_evaluate_contractions, sizeof(d_func_t));

  // Iterate through each chunk of the data set.
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  size_t i_iter = 0;
  while(index_to_copy < knumb_points) {
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
//    int3 numb_points_iter = {static_cast<int>(number_pts_iter), knumb_points.y, knumb_points.z};
    size_t total_numb_points_iter_bytes = number_pts_iter * sizeof(double);
    size_t total_contraction_arr_iter_bytes = total_numb_points_iter_bytes * t_nbasis;
    //printf("Number of points %zu \n", number_pts_iter);
    //printf("Maximal number of points in x-axis to do each chunk %zu \n", number_pts_iter);

    // Allocate device memory for contractions array, and set all elements to zero via cudaMemset.
    //    The contraction array rows are the atomic orbitals and columns are grid points and is stored in row-major order.
    double *d_contractions;
    cudaError err = cudaMalloc((double **) &d_contractions, total_contraction_arr_iter_bytes);
    // If error (most likely allocation error)
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating contractions, actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }
    chemtools::cuda_check_errors(cudaMemset(d_contractions, 0, total_contraction_arr_iter_bytes));

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

    // Evaluate contractions. The number of threads is maximal and the number of thread blocks is calculated.
    // Produces a matrix of size (N, M) where N is the number of points
    int ilen = 128;  // 128 320 1024
    dim3 threadsPerBlock(ilen);
    dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_contractions_func,
        d_contractions,
        d_points,
        number_pts_iter,
        nbasisfuncs,
        threadsPerBlock,
        grid
    );
    // Free the grid points in device memory.
    cudaFree(d_points);

    // Allocate device memory for the one_rdm. Number of atomic orbitals is equal to number of molecular obitals
    double *d_one_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, nbasisfuncs * nbasisfuncs * sizeof(double)));
    chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(),
                                                iodata.GetOneRdmShape(),
                                                sizeof(double),
                                                iodata.GetMOOneRDM(),
                                                iodata.GetOneRdmShape(),
                                                d_one_rdm,
                                                iodata.GetOneRdmShape()));

    // Allocate device memory for the array holding the matrix multiplication of one_rdm and contraction array.
    double *d_final;
    err = cudaMalloc((double **) &d_final, total_contraction_arr_iter_bytes);
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating d_final in evaluating density, "
                   "actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cudaFree(d_one_rdm);
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }

    // Matrix multiplcation of one rdm with the contractions array. Everything is in row major order.
    double alpha = 1.;
    double beta = 0.;
    chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            number_pts_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                            &alpha, d_contractions, number_pts_iter,
                                            d_one_rdm, iodata.GetOneRdmShape(), &beta,
                                            d_final, number_pts_iter));
    cudaFree(d_one_rdm); // one body reduced density matrix isn't needed anymore.

    // Do a hadamard product between d_final and d_contraction and store it in d_final. Maximal
    dim3 threadsPerBlock2(320);
    dim3 grid2((number_pts_iter * nbasisfuncs + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
    chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(d_final, d_contractions,
        nbasisfuncs, number_pts_iter);
    cudaFree(d_contractions);  // d_contractions is no longer needed.

    // Allocate device memory for electron density.
    double *d_electron_density;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_electron_density, total_numb_points_iter_bytes));

    // Sum up the columns of d_final to get the electron density. This is done by doing a matrix multiplcation of
    //    all ones of the transpose of d_final. Here I'm using the fact that d_final is in row major order.
    thrust::device_vector<double> all_ones(sizeof(double) * nbasisfuncs, 1.0);
    double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
    chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, number_pts_iter, nbasisfuncs,
                                            &alpha, d_final, number_pts_iter, deviceVecPtr, 1, &beta,
                                            d_electron_density, 1));
    cudaFree(d_final);

    // Transfer electron density from device memory to host memory.
    //    Since I'm computing a sub-grid at a time, need to update the index h_electron_density, accordingly.
    chemtools::cuda_check_errors(cudaMemcpy(&h_electron_density[0] + index_to_copy,
                                         d_electron_density,
                                         total_numb_points_iter_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_electron_density);

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.
  }
  return h_electron_density;
}


__host__ std::vector<double> chemtools::evaluate_molecular_orbitals_on_any_grid(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
){
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> density = chemtools::evaluate_molecular_orbitals_on_any_grid_handle(
      handle, iodata, h_points, knumb_points
  );
  cublasDestroy(handle); // cublas handle is no longer needed infact most of
  return density;
}


__host__ std::vector<double> chemtools::evaluate_molecular_orbitals_on_any_grid_handle(
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int knumb_points
) {
  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  int nbasisfuncs = molecular_basis.numb_basis_functions();

  // Calculate the number of bytes it would take memory of
  //   The highest memory operation is doing Y=XA, where X is contraction array, A is one-rdm
  //   You need to store two matrices of size (MxN) where M is number of contractions
  //   and N is the number of points and one matrix of size (MxM) for the one-rdm.
  //   Take 12 Gigabyte GPU, then convert it into bytes
  size_t t_numb_pts = knumb_points;
  size_t t_nbasis = nbasisfuncs;
  size_t t_highest_number_of_bytes = sizeof(double) * (2 * t_numb_pts * t_nbasis + t_nbasis * t_nbasis);
  size_t free_mem = 0;   // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  free_mem -= 500000000;
  // Calculate how much memory can fit inside a GPU memory.
  size_t t_numb_chunks = t_highest_number_of_bytes / free_mem;
  // Calculate how many points we can compute with 11.5 Gb:
  //    This is calculated by solving (2 * N M + M^2) * 8 bytes = 11.5 Gb(in bytes)  for N to get:
  size_t t_numb_pts_of_each_chunk = ((free_mem / (sizeof(double)))  - t_nbasis * t_nbasis) / (2 * t_nbasis);
  if (t_numb_pts_of_each_chunk == 0 and t_numb_chunks > 1.0) {
    // Haven't handle this case yet
    assert(0);
  }

  // Molecular orbitals shape (M, N), where M is number of orbitals and N is number of points, stored in col order
  std::vector<double> h_mol_orbitals_col(knumb_points * t_nbasis);

  // Function pointers and copy from device to host
  d_func_t h_contractions_func;
  cudaMemcpyFromSymbol(&h_contractions_func, chemtools::p_evaluate_contractions, sizeof(d_func_t));

  // Iterate through each chunk of the data set.
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  size_t i_iter = 0;
  while(index_to_copy < knumb_points) {
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
//    int3 numb_points_iter = {static_cast<int>(number_pts_iter), knumb_points.y, knumb_points.z};
    size_t total_numb_points_iter_bytes = number_pts_iter * sizeof(double);
    size_t total_contraction_arr_iter_bytes = total_numb_points_iter_bytes * t_nbasis;
    //printf("Number of points %zu \n", number_pts_iter);
    //printf("Maximal number of points in x-axis to do each chunk %zu \n", number_pts_iter);

    // Allocate device memory for contractions array, and set all elements to zero via cudaMemset.
    //    The contraction array rows are the atomic orbitals and columns are grid points and is stored in row-major order.
    double *d_contractions;
    cudaError err = cudaMalloc((double **) &d_contractions, total_contraction_arr_iter_bytes);
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating contractions, actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }
    chemtools::cuda_check_errors(cudaMemset(d_contractions, 0, total_contraction_arr_iter_bytes));

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

    // Evaluate contractions. The number of threads is maximal and the number of thread blocks is calculated.
    // Produces a matrix of size (N, M) where N is the number of points
    int ilen = 128;  // 128 320 1024
    dim3 threadsPerBlock(ilen);
    dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_contractions_func,
        d_contractions,
        d_points,
        number_pts_iter,
        nbasisfuncs,
        threadsPerBlock,
        grid
    );
    // Free the grid points in device memory.
    cudaFree(d_points);

    // Allocate device memory for the one_rdm. Number of atomic orbitals is equal to number of molecular obitals
    double *d_one_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, nbasisfuncs * nbasisfuncs * sizeof(double)));
    chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(),
                                                   iodata.GetOneRdmShape(),
                                                   sizeof(double),
                                                   iodata.GetMOOneRDM(),
                                                   iodata.GetOneRdmShape(),
                                                   d_one_rdm,
                                                   iodata.GetOneRdmShape()));

    // Allocate device memory for the array holding the matrix multiplication of one_rdm and contraction array.
    double *d_mol_orbs;
    err = cudaMalloc((double **) &d_mol_orbs, total_contraction_arr_iter_bytes);
    if (err != cudaSuccess) {
      std::cout << "Cuda Error in allocating d_final in evaluating density, "
                   "actual error: " << cudaGetErrorString(err) << std::endl;
      cudaFree(d_contractions);
      cudaFree(d_one_rdm);
      cudaFree(d_contractions);
      cublasDestroy(handle);
      throw err;
    }

    // Matrix multiplcation of one rdm with the contractions array. Everything is in row major order.
    double alpha = 1.;
    double beta = 0.;
    chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               number_pts_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                               &alpha, d_contractions, number_pts_iter,
                                               d_one_rdm, iodata.GetOneRdmShape(), &beta,
                                               d_mol_orbs, number_pts_iter));
    cudaFree(d_one_rdm); // one body reduced density matrix isn't needed anymore.


    // Transfer electron density from device memory to host memory.
    //    Since I'm computing a sub-grid at a time, need to update the index h_electron_density, accordingly.
    //    Note that d_mol_orbs is in row-major order with shape (M, N)
    std::vector<double> h_mol_orbitals_row(t_nbasis * number_pts_iter);
    chemtools::cuda_check_errors(cudaMemcpy(h_mol_orbitals_row.data(),
                                            d_mol_orbs,
                                            sizeof(double) * t_nbasis * number_pts_iter, cudaMemcpyDeviceToHost));
    cudaFree(d_mol_orbs);

    // Transfer from row to column major order;
    for(int i_row = 0; i_row < t_nbasis; i_row++){
      for(int j_col = 0; j_col < number_pts_iter; j_col++) {
        h_mol_orbitals_col[t_nbasis * index_to_copy + (j_col * t_nbasis + i_row)] = h_mol_orbitals_row[i_row * number_pts_iter + j_col];
      }
    }

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.
  }
  return h_mol_orbitals_col;
}
