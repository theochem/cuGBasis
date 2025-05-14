#include <stdio.h>
#include <thrust/device_vector.h>

#include "basis_to_gpu.cuh"
#include "cuda_basis_utils.cuh"
#include "cuda_utils.cuh"
#include "eval_esp.cuh"
#include "integral_coeffs.cuh"

/**
* @Section Computing the electrostatic potential.
* ----------------------------------------------
**/
// START COMPUTER GENERATED (USING PYTHON) CODE "generate_integrals.py"

__device__ void compute_row_s_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_s_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_s_px_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_s_py_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_s_pz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_s_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_s_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_s_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_s_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_s_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_s_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_s_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_s_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_s_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_s_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_s_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_s_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_s_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_s_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_s_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_s_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_s_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_s_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_s_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_s_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate s-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_s_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate s-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_s_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_s_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_s_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_s_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_s_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_s_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_s_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_s_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_s_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_s_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_s_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_s_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_s_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_s_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_s(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_s_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_s_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}

__device__ void compute_row_p_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_px_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_py_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_pz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_px_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_py_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_pz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-px
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_py_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate py-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_py_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_pz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-px
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_pz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate pz-py
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_pz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate pz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_pz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_px_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_px_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_px_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_px_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_px_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_px_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_py_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_py_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_py_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_py_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_py_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_py_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_pz_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_pz_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_pz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_pz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_pz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_pz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_px_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_px_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_px_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_px_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_px_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_py_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_py_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_py_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_py_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_py_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_pz_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_pz_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_pz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_pz_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_pz_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_px_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_px_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_px_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_px_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_px_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_px_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_px_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_px_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_px_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate px-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_px_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_py_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_py_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_py_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_py_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_py_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_py_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_py_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_py_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_py_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate py-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_py_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_pz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_pz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_pz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_pz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_pz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_pz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_pz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_pz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_pz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate pz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_pz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate px-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_px_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_px_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_px_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_px_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_px_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_px_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_px_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_px_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_px_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_px_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_px_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_px_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_px_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_px_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate px-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_px_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_px_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate py-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_py_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_py_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_py_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_py_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_py_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_py_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_py_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_py_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_py_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_py_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_py_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_py_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_py_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_py_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate py-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_py_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_py_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate pz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_pz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_pz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_pz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_pz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_pz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_pz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_pz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_pz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_pz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_pz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_pz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_pz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_pz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_pz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate pz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_p(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_pz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_pz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}

__device__ void compute_row_d_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxx-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxx-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dyy-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dyy-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dzz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dzz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxy-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxy-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dxz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dyz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate dyz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-dxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyy-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-dxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dzz-dyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dzz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dzz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dzz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dzz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-dxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxy-dyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxy-dzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxy-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-dxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxz-dyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxz-dzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxz-dxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dxz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-dxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyz-dyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyz-dzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyz-dxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyz-dxz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate dyz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dzz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dxy-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dxy-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dxz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dxz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dyz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dyz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate dyz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxx-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyy-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dzz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dxy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxy-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dxz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dxz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_dyz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_dyz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_dyz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate dyz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_dyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate dxx-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxx-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 2, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate dyy-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyy-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate dzz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dzz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate dxy-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxy-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate dxz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dxz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 1, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate dyz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate dyz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_d(alpha, 0, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}

__device__ void compute_row_dp_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               -0.5 * chemtools::compute_s_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_s_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_s_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               0.8660254037844386 * chemtools::compute_s_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_s_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               1.7320508075688772 * chemtools::compute_s_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.5 * chemtools::compute_px_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_px_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_px_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.5 * chemtools::compute_py_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_py_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_py_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.5 * chemtools::compute_pz_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_pz_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_pz_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.8660254037844386 * chemtools::compute_px_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_px_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c22-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.8660254037844386 * chemtools::compute_py_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_py_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c22-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.8660254037844386 * chemtools::compute_pz_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_pz_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_px_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_py_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.7320508075688772 * chemtools::compute_pz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               -0.5 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               -0.5 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               -0.5 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               -0.5 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               -0.5 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dyz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c22-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c22-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               0.8660254037844386 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s22-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * -0.5 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * -0.5 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.5 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * -0.5 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * 1 * chemtools::compute_dzz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * 1.7320508075688772 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1.7320508075688772 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 1.7320508075688772 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * 1.7320508075688772 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1.7320508075688772 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 1.7320508075688772 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * 0.8660254037844386 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * -0.8660254037844386 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 0.8660254037844386 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * -0.8660254037844386 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c20-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * 1.7320508075688772 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1.7320508075688772 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 1.7320508075688772 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-c20
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * -0.5 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.5 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * 1 * chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate c21-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 0.8660254037844386 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.8660254037844386 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c21-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-c20
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * -0.5 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.5 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * 1 * chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s21-c21
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s21-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dyz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 0.8660254037844386 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.8660254037844386 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s21-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-c20
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * -0.5 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * 1 * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.5 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * -0.5 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1 * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c22-c21
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c22-s21
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c22-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * 0.8660254037844386 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 0.8660254037844386 * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * -0.8660254037844386 * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1.7320508075688772 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-c20
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * -0.5 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.5 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * 1 * chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s22-c21
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s22-s21
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s22-c22
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 0.8660254037844386 * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.7320508075688772 * -0.8660254037844386 * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s22-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * 1.7320508075688772 * chemtools::compute_dxy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               -0.5 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               -0.5 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               -0.5 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               -0.5 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               -0.5 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               -0.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               -0.5 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               -0.5 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               -0.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               -0.5 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c20-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * 1.0 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1.0 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 1.0 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -1.5 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -1.5 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * -0.6123724356957945 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.449489742783178 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.449489742783178 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.6123724356957945 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.6123724356957945 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 2.449489742783178 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * -0.6123724356957945 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.449489742783178 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.6123724356957945 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.449489742783178 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.6123724356957945 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.6123724356957945 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 2.449489742783178 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * 1.9364916731037085 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.9364916731037085 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 1.9364916731037085 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -1.9364916731037085 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 1.9364916731037085 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -1.9364916731037085 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * 3.872983346207417 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 3.872983346207417 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 3.872983346207417 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * 0.7905694150420949 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -2.3717082451262845 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 0.7905694150420949 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -2.3717082451262845 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 0.7905694150420949 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -2.3717082451262845 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c20-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.5 * -0.7905694150420949 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.3717082451262845 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * -0.7905694150420949 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.5 * 2.3717082451262845 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * -0.7905694150420949 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1 * 2.3717082451262845 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c21-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.0 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.9364916731037085 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.9364916731037085 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 3.872983346207417 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 0.7905694150420949 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -2.3717082451262845 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c21-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.7905694150420949 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.3717082451262845 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s21-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.0 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.9364916731037085 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.9364916731037085 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 3.872983346207417 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 0.7905694150420949 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -2.3717082451262845 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s21-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.7905694150420949 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.3717082451262845 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c22-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * 1.0 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -1.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -1.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1.0 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -1.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -1.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * 2.449489742783178 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 2.449489742783178 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * 2.449489742783178 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.6123724356957945 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 2.449489742783178 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * 1.9364916731037085 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -1.9364916731037085 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 1.9364916731037085 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -1.9364916731037085 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * 3.872983346207417 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 3.872983346207417 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * 0.7905694150420949 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * -2.3717082451262845 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 0.7905694150420949 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -2.3717082451262845 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c22-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.8660254037844386 * -0.7905694150420949 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.8660254037844386 * 2.3717082451262845 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * -0.7905694150420949 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.8660254037844386 * 2.3717082451262845 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s22-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.0 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.5 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -0.6123724356957945 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.449489742783178 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 1.9364916731037085 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -1.9364916731037085 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 3.872983346207417 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * 0.7905694150420949 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * -2.3717082451262845 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s22-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_d(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.7320508075688772 * -0.7905694150420949 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.7320508075688772 * 2.3717082451262845 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}

__device__ void compute_row_f_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_s(beta) *
               chemtools::compute_s_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_px_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_py_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_p(beta) *
               chemtools::compute_pz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxx-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyy-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fzzz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyy-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxy-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxxz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxzz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyzz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fyyz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               chemtools::compute_dzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               chemtools::compute_dxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               chemtools::compute_dxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             // Integrate fxyz-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               chemtools::compute_dyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxx-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxx-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxx-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxx-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyy-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyy-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyy-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyy-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fzzz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fzzz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fzzz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fzzz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyy-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyy-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyy-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyy-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxy-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxy-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxy-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxy-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxzz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxzz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxzz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxzz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.5 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.5 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.8660254037844386 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.8660254037844386 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.7320508075688772 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxx-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyy-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyy-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fzzz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fzzz-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fzzz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyy-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyy-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyy-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxyy-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxy-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxy-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxy-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxy-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxy-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxy-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxy-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxy-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxy-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxz-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxz-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxz-fxxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxxz-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxxz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fxxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fxxz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxzz-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxzz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxzz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fxzz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fxxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fxxz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fxzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyzz-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fyzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyzz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyzz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fxxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fxxz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fxzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fyzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fyyz-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             // Integrate fyyz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-fxxx
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fyyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fzzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fxyy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fxxy
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fxxz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fxzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fyzz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fyyz
             if (icontr_shell != jcontr_shell) {
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P);
             }
             // Integrate fxyz-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               chemtools::compute_fxyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P);
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate fxxx-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxx-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 3, 0, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate fyyy-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyy-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 3, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate fzzz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fzzz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 0, 3) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate fxyy-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyy-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 2, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate fxxy-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxy-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxy-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxy-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxy-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxy-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxy-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 1, 0) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate fxxz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxxz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxxz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 2, 0, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate fxzz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxzz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxzz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxzz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxzz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxzz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxzz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 0, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 7) * knbasis - ((i_row + 7) * ((i_row + 7) - 1)) / 2) + j_col - (i_row + 7);// Move 1 row down.
             // Integrate fyzz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyzz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyzz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyzz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyzz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyzz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 1, 2) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 8) * knbasis - ((i_row + 8) * ((i_row + 8) - 1)) / 2) + j_col - (i_row + 8);// Move 1 row down.
             // Integrate fyyz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fyyz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fyyz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 0, 2, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 9) * knbasis - ((i_row + 9) * ((i_row + 9) - 1)) / 2) + j_col - (i_row + 9);// Move 1 row down.
             // Integrate fxyz-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * chemtools::compute_fxyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate fxyz-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate fxyz-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_f(alpha, 1, 1, 1) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}

__device__ void compute_row_fp_type_integral(const double3& A, const double3& pt,
    const int& numb_primitives1, double* d_point_charge,
    const int& point_index, int& i_row, const int& iconst, int& jconst,
    const int& knbasis, const int& npoints,
    const int& numb_contracted_shells, const int& icontr_shell,
    const double& screen_tol) {
  int i_integral;
  int j_col = i_row;
  // Enumerate through second basis set starting right after the contracted shell.
  for(int jcontr_shell = icontr_shell; jcontr_shell < numb_contracted_shells; jcontr_shell++) {
    double3 B = {g_constant_basis[jconst++], g_constant_basis[jconst++], g_constant_basis[jconst++]};
    int numb_primitives2 = (int) g_constant_basis[jconst++];
    int angmom_2 = (int) g_constant_basis[jconst++];
    // Enumerate through all primitives.
    for (int i_prim1 = 0; i_prim1 < numb_primitives1; i_prim1++) {
      double alpha = g_constant_basis[iconst + i_prim1];
      for (int i_prim2 = 0; i_prim2 < numb_primitives2; i_prim2++) {
        double beta = g_constant_basis[jconst + i_prim2];
        double3 P = {(alpha * A.x + beta * B.x) / (alpha + beta),
                     (alpha * A.y + beta * B.y) / (alpha + beta),
                     (alpha * A.z + beta * B.z) / (alpha + beta)};
        switch(angmom_2){
          case 0:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               1.0 * chemtools::compute_s_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_s_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_s_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               -0.6123724356957945 * chemtools::compute_s_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_s_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_s_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               -0.6123724356957945 * chemtools::compute_s_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_s_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_s_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               1.9364916731037085 * chemtools::compute_s_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_s_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               3.872983346207417 * chemtools::compute_s_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               0.7905694150420949 * chemtools::compute_s_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_s_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-s
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_s(beta) *
               (
               -0.7905694150420949 * chemtools::compute_s_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_s_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 1:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.0 * chemtools::compute_px_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_px_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_px_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.0 * chemtools::compute_py_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_py_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_py_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.0 * chemtools::compute_pz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_pz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_pz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_px_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_px_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_px_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_py_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_py_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_py_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_pz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_pz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_pz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_px_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_px_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_px_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_py_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_py_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_py_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.6123724356957945 * chemtools::compute_pz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_pz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_pz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.9364916731037085 * chemtools::compute_px_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_px_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.9364916731037085 * chemtools::compute_py_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_py_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               1.9364916731037085 * chemtools::compute_pz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_pz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               3.872983346207417 * chemtools::compute_px_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               3.872983346207417 * chemtools::compute_py_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               3.872983346207417 * chemtools::compute_pz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.7905694150420949 * chemtools::compute_px_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_px_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.7905694150420949 * chemtools::compute_py_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_py_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               0.7905694150420949 * chemtools::compute_pz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_pz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-px
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.7905694150420949 * chemtools::compute_px_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_px_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-py
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.7905694150420949 * chemtools::compute_py_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_py_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-pz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_p(beta) *
               (
               -0.7905694150420949 * chemtools::compute_pz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_pz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               1.0 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               1.0 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               1.0 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               1.0 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               1.0 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               1.0 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               -0.6123724356957945 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               -0.6123724356957945 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               -0.6123724356957945 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               -0.6123724356957945 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               -0.6123724356957945 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               -0.6123724356957945 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               -0.6123724356957945 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               1.9364916731037085 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               1.9364916731037085 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               1.9364916731037085 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               1.9364916731037085 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               1.9364916731037085 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               1.9364916731037085 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               3.872983346207417 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               3.872983346207417 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               3.872983346207417 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               3.872983346207417 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               3.872983346207417 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               3.872983346207417 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               0.7905694150420949 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               0.7905694150420949 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               0.7905694150420949 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               0.7905694150420949 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               0.7905694150420949 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               0.7905694150420949 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-dxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 2, 0, 0) *
               (
               -0.7905694150420949 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-dyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 2, 0) *
               (
               -0.7905694150420949 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-dzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 0, 2) *
               (
               -0.7905694150420949 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-dxy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 1, 0) *
               (
               -0.7905694150420949 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-dxz
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 1, 0, 1) *
               (
               -0.7905694150420949 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-dyz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_d(beta, 0, 1, 1) *
               (
               -0.7905694150420949 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case -2:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.0 * -0.5 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * -0.5 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * 1 * chemtools::compute_dzz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.0 * 1.7320508075688772 * chemtools::compute_dxz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.0 * 1.7320508075688772 * chemtools::compute_dyz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.0 * 0.8660254037844386 * chemtools::compute_dxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * -0.8660254037844386 * chemtools::compute_dyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 0.8660254037844386 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.8660254037844386 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 0.8660254037844386 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.8660254037844386 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.0 * 1.7320508075688772 * chemtools::compute_dxy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 1.7320508075688772 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * -0.5 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.5 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.5 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1 * chemtools::compute_dzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dyz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 0.8660254037844386 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.8660254037844386 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 0.8660254037844386 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.8660254037844386 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 0.8660254037844386 * chemtools::compute_dxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.8660254037844386 * chemtools::compute_dyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * -0.5 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.5 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.5 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.5 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1 * chemtools::compute_dzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dyz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 0.8660254037844386 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.8660254037844386 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 0.8660254037844386 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.8660254037844386 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 0.8660254037844386 * chemtools::compute_dxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.8660254037844386 * chemtools::compute_dyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 1.7320508075688772 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 1.7320508075688772 * chemtools::compute_dxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.9364916731037085 * -0.5 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -0.5 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * 1 * chemtools::compute_dzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.5 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.5 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 1 * chemtools::compute_dzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dyz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.9364916731037085 * 0.8660254037844386 * chemtools::compute_dxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -0.8660254037844386 * chemtools::compute_dyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 0.8660254037844386 * chemtools::compute_dxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.8660254037844386 * chemtools::compute_dyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 1.7320508075688772 * chemtools::compute_dxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               3.872983346207417 * -0.5 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -0.5 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * 1 * chemtools::compute_dzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               3.872983346207417 * 1.7320508075688772 * chemtools::compute_dxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               3.872983346207417 * 1.7320508075688772 * chemtools::compute_dyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               3.872983346207417 * 0.8660254037844386 * chemtools::compute_dxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -0.8660254037844386 * chemtools::compute_dyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               3.872983346207417 * 1.7320508075688772 * chemtools::compute_dxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.7905694150420949 * -0.5 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 0.7905694150420949 * -0.5 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 0.7905694150420949 * 1 * chemtools::compute_dzz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -0.5 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -0.5 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 1 * chemtools::compute_dzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dxz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dxz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dyz_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dyz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.7905694150420949 * 0.8660254037844386 * chemtools::compute_dxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 0.7905694150420949 * -0.8660254037844386 * chemtools::compute_dyy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 0.8660254037844386 * chemtools::compute_dxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -0.8660254037844386 * chemtools::compute_dyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dxy_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dxy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-c20
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.7905694150420949 * -0.5 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.7905694150420949 * -0.5 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.7905694150420949 * 1 * chemtools::compute_dzz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -0.5 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -0.5 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 1 * chemtools::compute_dzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-c21
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dxz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dxz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-s21
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dyz_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dyz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-c22
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.7905694150420949 * 0.8660254037844386 * chemtools::compute_dxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.7905694150420949 * -0.8660254037844386 * chemtools::compute_dyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 0.8660254037844386 * chemtools::compute_dxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -0.8660254037844386 * chemtools::compute_dyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-s22
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_d(beta) *
               (
               -0.7905694150420949 * 1.7320508075688772 * chemtools::compute_dxy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 1.7320508075688772 * chemtools::compute_dxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             break;
          case 3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               1.0 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               1.0 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               1.0 * chemtools::compute_fzzz_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               1.0 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               1.0 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               1.0 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               1.0 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               1.0 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               1.0 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               1.0 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s31-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               -0.6123724356957945 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               1.9364916731037085 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               1.9364916731037085 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               1.9364916731037085 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               1.9364916731037085 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               1.9364916731037085 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               1.9364916731037085 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               3.872983346207417 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               3.872983346207417 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               3.872983346207417 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               3.872983346207417 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               3.872983346207417 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               3.872983346207417 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               3.872983346207417 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               3.872983346207417 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               3.872983346207417 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               3.872983346207417 * chemtools::compute_fxyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c33-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               0.7905694150420949 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-fxxx
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 3, 0, 0) *
               (
               -0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-fyyy
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 3, 0) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-fzzz
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 0, 3) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-fxyy
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 2, 0) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-fxxy
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 1, 0) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s33-fxxz
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 2, 0, 1) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s33-fxzz
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 0, 2) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s33-fyzz
             d_point_charge[point_index + (i_integral + 7) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 1, 2) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s33-fyyz
             d_point_charge[point_index + (i_integral + 8) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 0, 2, 1) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s33-fxyz
             d_point_charge[point_index + (i_integral + 9) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_f(beta, 1, 1, 1) *
               (
               -0.7905694150420949 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
          case -3:
             i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
             // Integrate c30-c30
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * 1.0 * chemtools::compute_fzzz_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.0 * -1.5 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.0 * -1.5 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 1.0 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -1.5 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 1.0 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -1.5 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c30-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * -0.6123724356957945 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * -0.6123724356957945 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.0 * 2.449489742783178 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.449489742783178 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.449489742783178 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * -0.6123724356957945 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * -0.6123724356957945 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.0 * 2.449489742783178 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.449489742783178 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.6123724356957945 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.449489742783178 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * 1.9364916731037085 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.0 * -1.9364916731037085 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 1.9364916731037085 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -1.9364916731037085 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c30-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * 3.872983346207417 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 3.872983346207417 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 3.872983346207417 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c30-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * 0.7905694150420949 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * -2.3717082451262845 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * 0.7905694150420949 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -2.3717082451262845 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 0.7905694150420949 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -2.3717082451262845 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c30-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.0 * -0.7905694150420949 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.0 * 2.3717082451262845 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.5 * -0.7905694150420949 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.3717082451262845 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * -0.7905694150420949 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.5 * 2.3717082451262845 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col - (i_row + 1);// Move 1 row down.
             // Integrate c31-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 1.0 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 1.0 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 1.0 * chemtools::compute_fzzz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.5 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.5 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c31-c31
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.449489742783178 * chemtools::compute_fxzz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 1.9364916731037085 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.9364916731037085 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 1.9364916731037085 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.9364916731037085 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 1.9364916731037085 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.9364916731037085 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 3.872983346207417 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 3.872983346207417 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 3.872983346207417 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c31-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 0.7905694150420949 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -2.3717082451262845 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 0.7905694150420949 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -2.3717082451262845 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 0.7905694150420949 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -2.3717082451262845 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c31-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.3717082451262845 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.7905694150420949 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.7905694150420949 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.3717082451262845 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col - (i_row + 2);// Move 1 row down.
             // Integrate s31-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 1.0 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 1.0 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.5 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 1.0 * chemtools::compute_fzzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.5 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.5 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s31-c31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.449489742783178 * chemtools::compute_fxzz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s31-s31
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -0.6123724356957945 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.449489742783178 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -0.6123724356957945 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.449489742783178 * chemtools::compute_fyzz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s31-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 1.9364916731037085 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.9364916731037085 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 1.9364916731037085 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -1.9364916731037085 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 1.9364916731037085 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -1.9364916731037085 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s31-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 3.872983346207417 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 3.872983346207417 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * 3.872983346207417 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s31-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * 0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -2.3717082451262845 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 0.7905694150420949 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * -2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 0.7905694150420949 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * -2.3717082451262845 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s31-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.6123724356957945 * -0.7905694150420949 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * 2.3717082451262845 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.6123724356957945 * -0.7905694150420949 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.6123724356957945 * 2.3717082451262845 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.449489742783178 * -0.7905694150420949 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.449489742783178 * 2.3717082451262845 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col - (i_row + 3);// Move 1 row down.
             // Integrate c32-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * 1.0 * chemtools::compute_fzzz_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -1.5 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.9364916731037085 * -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * 1.0 * chemtools::compute_fzzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -1.5 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -1.5 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c32-c31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * 2.449489742783178 * chemtools::compute_fxxz_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 2.449489742783178 * chemtools::compute_fxzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate c32-s31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * 2.449489742783178 * chemtools::compute_fxxz_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.6123724356957945 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 2.449489742783178 * chemtools::compute_fyzz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate c32-c32
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * 1.9364916731037085 * chemtools::compute_fxxz_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 1.9364916731037085 * -1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * 1.9364916731037085 * chemtools::compute_fxxz_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -1.9364916731037085 * chemtools::compute_fyyz_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c32-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * 3.872983346207417 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -1.9364916731037085 * 3.872983346207417 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c32-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * 0.7905694150420949 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * -2.3717082451262845 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 0.7905694150420949 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -2.3717082451262845 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate c32-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               1.9364916731037085 * -0.7905694150420949 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 1.9364916731037085 * 2.3717082451262845 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * -0.7905694150420949 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -1.9364916731037085 * 2.3717082451262845 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col - (i_row + 4);// Move 1 row down.
             // Integrate s32-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * 1.0 * chemtools::compute_fzzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -1.5 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -1.5 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s32-c31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * -0.6123724356957945 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -0.6123724356957945 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * 2.449489742783178 * chemtools::compute_fxzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s32-s31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * -0.6123724356957945 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -0.6123724356957945 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * 2.449489742783178 * chemtools::compute_fyzz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s32-c32
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * 1.9364916731037085 * chemtools::compute_fxxz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -1.9364916731037085 * chemtools::compute_fyyz_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s32-s32
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * 3.872983346207417 * chemtools::compute_fxyz_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate s32-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * 0.7905694150420949 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * -2.3717082451262845 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             // Integrate s32-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               3.872983346207417 * -0.7905694150420949 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 3.872983346207417 * 2.3717082451262845 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             i_integral = ((i_row + 5) * knbasis - ((i_row + 5) * ((i_row + 5) - 1)) / 2) + j_col - (i_row + 5);// Move 1 row down.
             // Integrate c33-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * 1.0 * chemtools::compute_fxxx_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -1.5 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -1.5 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 1.0 * chemtools::compute_fzzz_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -1.5 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * -1.5 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c33-c31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * 2.449489742783178 * chemtools::compute_fxxx_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 2.449489742783178 * chemtools::compute_fxyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c33-s31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * 2.449489742783178 * chemtools::compute_fxxx_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 2.449489742783178 * chemtools::compute_fxyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c33-c32
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * 1.9364916731037085 * chemtools::compute_fxxx_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -1.9364916731037085 * chemtools::compute_fxxx_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 1.9364916731037085 * chemtools::compute_fxyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * -1.9364916731037085 * chemtools::compute_fxyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c33-s32
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * 3.872983346207417 * chemtools::compute_fxxx_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 3.872983346207417 * chemtools::compute_fxyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate c33-c33
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * 0.7905694150420949 * chemtools::compute_fxxx_fxxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * -2.3717082451262845 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * 0.7905694150420949 * chemtools::compute_fxxx_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * -2.3717082451262845 * chemtools::compute_fxyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             // Integrate c33-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               0.7905694150420949 * -0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 0.7905694150420949 * 2.3717082451262845 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -2.3717082451262845 * -0.7905694150420949 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -2.3717082451262845 * 2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             i_integral = ((i_row + 6) * knbasis - ((i_row + 6) * ((i_row + 6) - 1)) / 2) + j_col - (i_row + 6);// Move 1 row down.
             // Integrate s33-c30
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 0) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * 1.0 * chemtools::compute_fyyy_fzzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * -1.5 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * -1.5 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * 1.0 * chemtools::compute_fzzz_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -1.5 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * -1.5 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s33-c31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 1) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * 2.449489742783178 * chemtools::compute_fyyy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 2.449489742783178 * chemtools::compute_fxxy_fxzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s33-s31
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 2) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * 2.449489742783178 * chemtools::compute_fyyy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -0.6123724356957945 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * 2.449489742783178 * chemtools::compute_fxxy_fyzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s33-c32
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 3) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * 1.9364916731037085 * chemtools::compute_fyyy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * -1.9364916731037085 * chemtools::compute_fyyy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * 1.9364916731037085 * chemtools::compute_fxxy_fxxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * -1.9364916731037085 * chemtools::compute_fxxy_fyyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s33-s32
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 4) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * 3.872983346207417 * chemtools::compute_fyyy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * 3.872983346207417 * chemtools::compute_fxxy_fxyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             }
             // Integrate s33-c33
             if (jcontr_shell != icontr_shell) {
             d_point_charge[point_index + (i_integral + 5) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * 0.7905694150420949 * chemtools::compute_fxxx_fyyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + -0.7905694150420949 * -2.3717082451262845 * chemtools::compute_fyyy_fxyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * 0.7905694150420949 * chemtools::compute_fxxx_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * -2.3717082451262845 * chemtools::compute_fxyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
               );
             }
             // Integrate s33-s33
             d_point_charge[point_index + (i_integral + 6) * npoints] +=
               g_constant_basis[iconst + numb_primitives1 + i_prim1] *
               g_constant_basis[jconst + numb_primitives2 + i_prim2] *
               chemtools::normalization_primitive_pure_f(alpha) *
               chemtools::normalization_primitive_pure_f(beta) *
               (
               -0.7905694150420949 * -0.7905694150420949 * chemtools::compute_fyyy_fyyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + -0.7905694150420949 * 2.3717082451262845 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                + 2.3717082451262845 * -0.7905694150420949 * chemtools::compute_fyyy_fxxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                + 2.3717082451262845 * 2.3717082451262845 * chemtools::compute_fxxy_fxxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
               );
             break;
        } // End switch
       }// End primitive 2
    }// End primitive 1
    // Update index to go to the next segmented shell.
    switch(angmom_2){
      case 0: j_col += 1;
        break;
      case 1: j_col += 3;
        break;
      case 2: j_col += 6;
        break;
      case 3: j_col += 10;
        break;
      case -2: j_col += 5;
        break;
      case -3: j_col += 7;
        break;
    } // End switch
    // Update index of constant memory to the next contracted shell of second basis set.
    jconst += 2 * numb_primitives2;
  }// End contracted shell 2
}





// END COMPUTER GENERATED (USING PYTHON) CODE "generate_integrals.py"


/**
 * TODO: Make the requirement that contracted shells are segmented to reduce the number of for loops.
 */
__global__ void chemtools::compute_point_charge_integrals(
    double* d_point_charge, const double* const d_grid, const int knumb_pts, const int nbasis, const double screen_tol) {
  // Every thread correspond to a grid point, the following is the mapping.
  int point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index < knumb_pts) {
    // Setup the initial variables.
    double3 pt = {d_grid[point_index], d_grid[point_index + knumb_pts], d_grid[point_index + 2 * knumb_pts]};
    int iconst = 0;                                                          // Index to go over constant memory.
    const int numb_contracted_shells = (int) g_constant_basis[iconst++];     // Number of contracted shell.
    const int knbasis = nbasis;                                              // Used to update different rows.
    int i_row = 0;                                                           // Index to iterate through rows
    // Enumerate through first basis set.
    for(int icontr_shell = 0; icontr_shell < numb_contracted_shells; icontr_shell++) {
      // Second index to go over constant memory. Due to symmetry of integrals, only need to go beyond iconst.
      int jconst = iconst;

      double3 A = {g_constant_basis[iconst++], g_constant_basis[iconst++], g_constant_basis[iconst++]};
      int numb_primitives_1 = (int) g_constant_basis[iconst++];
      int angmom_1 = (int) g_constant_basis[iconst++];
      if (angmom_1 == 0) {
        // These are the rows S-type (one row only)
        // Just enumerate through everything, updating "ipoint_charge" which tells you where in the 2d array you are.
        // Enumerating means that I'm going through all second basis-set right after this one and updating jconst.
        // ipoint_charge is updated based on the second basis-set.
        compute_row_s_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
        );
      }
      else if (angmom_1 == 1) {
        compute_row_p_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == 2) {
        compute_row_d_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == 3) {
        compute_row_f_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == -2) {
        compute_row_dp_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == -3) {
        compute_row_fp_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == -4) {
        std::printf("G-type orbitals is not supported yet");
        assert(1 == 0);
      }
      // Update index to go to the next segmented shell.
      switch(angmom_1){
        case 0: i_row += 1;
          break;
        case 1: i_row += 3;
          break;
        case 2: i_row += 6;
          break;
        case 3: i_row += 10;
          break;
        case -2: i_row += 5;
          break;
        case -3: i_row += 7;
          break;
      } // End switch
      // Update index of constant memory to the next contracted shell.
      iconst += 2 * numb_primitives_1;
    }  // End first basis set.
  } // End thread if condition
}


__host__ std::vector<double> chemtools::compute_electrostatic_potential_over_points(
    chemtools::IOData& iodata, double* grid, int knumb_pts, const double screen_tol, const bool disp, const std::string& spin) {
  // Place it into constant memory. The second argument must be true, as the decontracted basis set must be used.
  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  int nbasisfuncs = molecular_basis.numb_basis_functions();

  // Return index
  std::array<std::size_t, 2> const_mem = chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, true);
  std::size_t index_shell = const_mem[0];
  if (index_shell != molecular_basis.numb_contracted_shells()) {
    std::string err_message2 = "This only works for basis-set that can fit entirely into constant memory. \n"
                               "Index of the shell " + std::to_string(index_shell) +
                               " is equal to number of contracted shells: " +
        std::to_string(molecular_basis.numb_contracted_shells());
    throw std::runtime_error(err_message2);
  }

  // Define the variables for the sizes in integer or in bytes.
  size_t t_numb_pts = knumb_pts;
  size_t t_nbasis = nbasisfuncs;
  size_t t_dim_rdm = t_nbasis * t_nbasis;
  size_t t_total_numb_integrals = t_numb_pts * t_nbasis * (t_nbasis + 1) / 2;
  size_t t_numb_pts_bytes = t_numb_pts * 3 * sizeof(double);                              // Each point has three double.
  size_t t_total_size_integrals_bytes = sizeof(double) * t_total_numb_integrals;
  size_t t_total_bytes = t_numb_pts_bytes + t_total_size_integrals_bytes * 2;             // Store grid and integrals twice.
  if (disp) {
    printf("Total Basis functions %zu \n", nbasisfuncs);
    printf("Basis Set Squared (RDM size) %zu \n", t_dim_rdm);
    printf("Number of integrals of all points %zu \n ", t_total_numb_integrals);
    printf("Total number of points %zu \n", t_numb_pts);
  }
  // Create the handles for using cublas.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Allocate one_rdm, identity matrix and the intermediate array called d_final. This is fixed throughout.
  double *d_one_rdm, *d_identity, *d_intermed;
  chemtools::cuda_check_errors(cudaMalloc((double **)&d_one_rdm, sizeof(double) * t_dim_rdm));
  chemtools::cuda_check_errors(cudaMalloc((double **)&d_intermed, sizeof(double) * t_dim_rdm));
  chemtools::cuda_check_errors(cudaMalloc((double **)&d_identity, sizeof(double) * t_dim_rdm));
  if (nbasisfuncs > 1024) {
    std::runtime_error("Number of basis functions is greater than maximum thread per block. Just need to change this");
  }
  dim3 blockDim(nbasisfuncs, nbasisfuncs);
  dim3 gridDim (1, 1);
  chemtools::set_identity_row_major<<<blockDim, gridDim>>>(d_identity, nbasisfuncs, nbasisfuncs);
  CUBLAS_CHECK(cublasSetMatrix (t_nbasis, t_nbasis, sizeof(double), iodata.GetMOOneRDM(spin),
                                               t_nbasis, d_one_rdm, t_nbasis));
  //printf("Print ideti t\n");
  //print_firstt_ten_elements<<<1, 1>>>(d_identity);
  //printf("print  one_Rdm\n");
  //print_firstt_ten_elements<<<1, 1>>>(d_one_rdm);
  //cudaDeviceSynchronize();

  // Set cache perference to L1 and compute the point charge integrals.
  cudaFuncSetCacheConfig(chemtools::compute_point_charge_integrals, cudaFuncCachePreferL1);
  // Allocate electrostatic vector that gets returned.
  std::vector<double> electrostatic(knumb_pts);

  // Calculate how much memory can fit inside GPU memory.
  size_t free_mem = 0;   // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  free_mem -= 500000000;  // Substract 0.5 Gb for safe measures
  size_t t_numb_chunks = t_total_bytes / free_mem;
  // Maximal number of points to do each iteration to achieve 11 GB of GPU memory.
  size_t t_numb_pts_of_each_chunk = static_cast<size_t>(((free_mem / sizeof(double)) - (5.0 * t_nbasis * t_nbasis)) / (1.5 * t_nbasis * (t_nbasis + 1)));
  //size_t t_numb_pts_of_each_chunk = 11000000000  / (3 * 8 + 8 * t_nbasis * (t_nbasis + 1));  // Solving 11Gb = Number of Pts * 3 * 8 + 2 * (number of integrals)
  size_t t_index_of_grid_at_ith_iter = 0;
  if (disp) {
    printf("Number of chunks %zu \n", t_numb_chunks);
    printf("Maximal number of points to do each chunk %zu \n", t_numb_pts_of_each_chunk);
  }
  double *d_grid, *d_point_charge, *d_point_charge_transpose;
  double alpha = 1.; double beta = 0.;
  // Iterate through each chunk of the data set.
  //for(size_t i_iter = 0; i_iter < t_numb_chunks + 1; i_iter++) {
  //for(size_t i_iter = t_numb_chunks + 1; i_iter <= t_numb_chunks + 1; i_iter++) {
  size_t i_iter = 0;
  while(t_index_of_grid_at_ith_iter < t_numb_pts) {
    // Figure out how many points to do this iteration.
    size_t t_numb_pts_ith_iter = std::min(t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk);
    if (disp) {
      printf("ITH CHUNK %zu and Number of points in ith %zu \n", i_iter, t_numb_pts_ith_iter);
    }
    size_t t_numb_pts_ith_iter_bytes = t_numb_pts_ith_iter * 3 * sizeof(double);    // Calculate number of bytes in this iter.
    size_t t_total_numb_integrals_ith_iter = t_numb_pts_ith_iter * t_nbasis * (t_nbasis + 1) / 2;
    size_t t_total_size_integrals_ith_iter_bytes = sizeof(double) * t_total_numb_integrals_ith_iter;

    // Transfer portion of the grid in row-major order into column-major order stored in portion_grid_col.
    double* portion_grid_col = new double[t_numb_pts_ith_iter * 3];
    int counter = 0;
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < t_numb_pts_ith_iter; j++) {
        portion_grid_col[counter] = grid[i + (t_index_of_grid_at_ith_iter + j) * 3];
        counter += 1;
      }
    }

    // Transfer portion of the grid to GPU, this grid needs to be stored in Column-major order.
    chemtools::cuda_check_errors(cudaMalloc((double **)&d_grid, t_numb_pts_ith_iter_bytes));
    chemtools::cuda_check_errors(cudaMemcpy(d_grid, portion_grid_col, t_numb_pts_ith_iter_bytes,
                                         cudaMemcpyHostToDevice));
    delete[] portion_grid_col;
    printf("Print grid \n");
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_grid);
    cudaDeviceSynchronize();

    // Allocate one-point charge integral array, stored as integrals then points, and set it to zero.
    chemtools::cuda_check_errors(cudaMalloc((double **)&d_point_charge, t_total_size_integrals_ith_iter_bytes));
    chemtools::cuda_check_errors(cudaMemset(d_point_charge, 0, t_total_size_integrals_ith_iter_bytes));
    // Compute point charge integrals.
    if (disp) {
      printf("Compute point charge integrals\n");
    }
    dim3 threadsPerBlock32(96); // 128, 96 same as 320 speed, 1024 one second slower, 64 is really slow.
    // 256 -> 40 seconds
    // 64  -> 27 seconds
    // 1024 -> 132 seconds
    // 32 -> 29 seconds
    // 128 -> 28 seconds
    // 96 -> 28 seconds
    //dim3 grid32 ((t_total_numb_integrals_ith_iter + threadsPerBlock32.x - 1) / (threadsPerBlock32.x));
    dim3 grid32 ((t_numb_pts_ith_iter + threadsPerBlock32.x - 1) / (threadsPerBlock32.x));
    if (disp) {
      printf("Number of threads %d \n", threadsPerBlock32.x);
      printf("Grid size %d \n", grid32.x);
    }
    chemtools::compute_point_charge_integrals<<<grid32, threadsPerBlock32>>>(
        d_point_charge, d_grid, (int) t_numb_pts_ith_iter, nbasisfuncs, screen_tol
        );
    cudaDeviceSynchronize();
    printf("Print d_point charge \n");
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_point_charge);
//    chemtools::print_all<<<1, 1>>>(d_point_charge, t_nbasis * (t_nbasis + 1) / 2);
    cudaDeviceSynchronize();
    // Free Grid in Device
    cudaFree(d_grid);

    // Transpose point_charge from (Z, Y, X) (col-major) to (Y, X, Z), where Z=number of points, Y, X are the contractions.
    cudaDeviceSynchronize();
    chemtools::cuda_check_errors(cudaMalloc((double **)&d_point_charge_transpose, t_total_size_integrals_ith_iter_bytes));
    printf("Dp d_point charge transpose \n");
    cudaDeviceSynchronize();
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                            nbasisfuncs * (nbasisfuncs + 1) / 2, (int) t_numb_pts_ith_iter,
                                            &alpha, d_point_charge, (int) t_numb_pts_ith_iter,
                                            &beta, d_point_charge, (int) t_numb_pts_ith_iter,
                                            d_point_charge_transpose, nbasisfuncs * (nbasisfuncs + 1) / 2));
    printf("Print d point charge transpose \n");
    //print_final_ten_elements<<<1, 1>>>(d_point_charge_transpose, t_total_numb_integrals_ith_iter);
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_point_charge_transpose);
    cudaDeviceSynchronize();
    //print_firstt_ten_elements<<<1, 1>>>(d_point_charge_transpose);

    // Free up d_point_charge
    cudaFree(d_point_charge);
    // The point charge array must be converted from  triangular packed format to
    //  triangular format for matrix-multiplcation.
    double* d_triangular_format;
    chemtools::cuda_check_errors(cudaMalloc((double **)&d_triangular_format, sizeof(double) * t_dim_rdm));
    chemtools::cuda_check_errors(cudaMemset(d_triangular_format, 0, sizeof(double) * t_dim_rdm));
    // Go through each grid point and calculate one component of the electrostatic potential.
    for(size_t i = 0; i < t_numb_pts_ith_iter; i++) {
      //  Conversion from the triangular packed format to the triangular format
      CUBLAS_CHECK(cublasDtpttr(handle, CUBLAS_FILL_MODE_LOWER,
                                               nbasisfuncs,
                                               &d_point_charge_transpose[i * t_nbasis * (t_nbasis + 1) / 2],
                                               d_triangular_format,
                                               nbasisfuncs
      ));
      // Symmetric Matrix multiplication by the identity matrix to convert it to a full-matrix
      //  TODO: THIS PART IS SLOW, IT COULD BE SOMEHOW REMOVED>
      CUBLAS_CHECK(cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                              nbasisfuncs, nbasisfuncs,
                                              &alpha, d_triangular_format, nbasisfuncs,  // A
                                              d_identity, nbasisfuncs,                   // B
                                              &beta, d_intermed, nbasisfuncs));
      // Hadamard Product with One RDM
      dim3 threadsPerBlock2(320);
      dim3 grid2 ((t_nbasis * t_nbasis + threadsPerBlock2.x - 1) / (threadsPerBlock2.x));
      chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(d_intermed, d_one_rdm, nbasisfuncs, nbasisfuncs);

      // Sum over the entire matrix to get the final point_charge.
      thrust::device_ptr<double> deviceVecPtr = thrust::device_pointer_cast(d_intermed);
      electrostatic[t_index_of_grid_at_ith_iter + i] = thrust::reduce(
          deviceVecPtr, deviceVecPtr + nbasisfuncs * nbasisfuncs
      );
    }
    // Update index for the next iteration/chunk of grid.
    t_index_of_grid_at_ith_iter += t_numb_pts_ith_iter;
    i_iter += 1;

    cudaFree(d_point_charge_transpose);
    cudaFree(d_triangular_format);
  }
  CUBLAS_CHECK(cublasDestroy(handle));  // cublas handle is no longer needed infact most of
  cudaFree(d_one_rdm);
  cudaFree(d_identity);
  cudaFree(d_intermed);
  // Subtract by the charge of the nucleus to get the final result.
  for(int i = 0; i < knumb_pts; i++){
    electrostatic[i] *= (-1.);
    for(int j = 0; j < iodata.GetNatoms(); j++) {
      double val = iodata.GetCharges()[j] /
//          std::sqrt(std::pow(grid[i] - iodata.GetCoordAtoms()[j * 3], 2) +
//              std::pow(grid[i + knumb_pts] - iodata.GetCoordAtoms()[j* 3 + 1], 2) +
//              std::pow(grid[i + (knumb_pts * 2)] - iodata.GetCoordAtoms()[j * 3 + 2], 2) );
          std::sqrt(std::pow(grid[i * 3] - iodata.GetCoordAtoms()[j * 3], 2) +
              std::pow(grid[i * 3 + 1] - iodata.GetCoordAtoms()[j* 3 + 1], 2) +
              std::pow(grid[i * 3 + 2] - iodata.GetCoordAtoms()[j * 3 + 2], 2) );
      electrostatic[i] += val;
    }
  }
  return electrostatic;
}
