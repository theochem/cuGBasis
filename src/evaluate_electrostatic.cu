#include <stdio.h>
#include <thrust/device_vector.h>

#include "../include/basis_to_gpu.cuh"
#include "../include/cuda_basis_utils.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/evaluate_electrostatic.cuh"
#include "../include/integral_coeffs.cuh"

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
        switch(angmom_2) {
          case 0:i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
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
            i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col
                - (i_row + 1);// Move 1 row down.
            // Integrate c21-s
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_s(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_s_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col
                - (i_row + 2);// Move 1 row down.
            // Integrate s21-s
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_s(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_s_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col
                - (i_row + 3);// Move 1 row down.
            // Integrate c22-s
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_s(beta) *
                    (
                        0.8660254037844386
                            * chemtools::compute_s_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_s_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col
                - (i_row + 4);// Move 1 row down.
            // Integrate s22-s
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_s(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_s_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            break;
          case 1:i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
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
            i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col
                - (i_row + 1);// Move 1 row down.
            // Integrate c21-px
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_px_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-py
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_py_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-pz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_pz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col
                - (i_row + 2);// Move 1 row down.
            // Integrate s21-px
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_px_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-py
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_py_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-pz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_pz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col
                - (i_row + 3);// Move 1 row down.
            // Integrate c22-px
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        0.8660254037844386
                            * chemtools::compute_px_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_px_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c22-py
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        0.8660254037844386
                            * chemtools::compute_py_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_py_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c22-pz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        0.8660254037844386
                            * chemtools::compute_pz_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_pz_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col
                - (i_row + 4);// Move 1 row down.
            // Integrate s22-px
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_px_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-py
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_py_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-pz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_p(beta) *
                    (
                        1.7320508075688772
                            * chemtools::compute_pz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            break;
          case 2:i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
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
            i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col
                - (i_row + 1);// Move 1 row down.
            // Integrate c21-dxx
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 2, 0, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-dyy
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 2, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-dzz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 0, 2) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-dxy
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 1, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-dxz
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 0, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-dyz
            d_point_charge[point_index + (i_integral + 5) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 1, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col
                - (i_row + 2);// Move 1 row down.
            // Integrate s21-dxx
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 2, 0, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-dyy
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 2, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-dzz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 0, 2) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-dxy
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 1, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-dxz
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 0, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-dyz
            d_point_charge[point_index + (i_integral + 5) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 1, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dyz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col
                - (i_row + 3);// Move 1 row down.
            // Integrate c22-dxx
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 2, 0, 0) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dxx_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c22-dyy
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 2, 0) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dyy_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c22-dzz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 0, 2) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c22-dxy
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 1, 0) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c22-dxz
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 0, 1) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c22-dyz
            d_point_charge[point_index + (i_integral + 5) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 1, 1) *
                    (
                        0.8660254037844386
                            * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386
                                * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col
                - (i_row + 4);// Move 1 row down.
            // Integrate s22-dxx
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 2, 0, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-dyy
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 2, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-dzz
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 0, 2) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-dxy
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 1, 0) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s22-dxz
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 1, 0, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate s22-dyz
            d_point_charge[point_index + (i_integral + 5) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_d(beta, 0, 1, 1) *
                    (
                        1.7320508075688772
                            * chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            break;
          case -2:i_integral = (i_row * knbasis - (i_row * (i_row - 1)) / 2) + j_col - i_row;// Move 1 row down.
            // Integrate c20-c20
            d_point_charge[point_index + (i_integral + 0) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        -0.5 * -0.5 * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * -0.5
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 1
                                * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * -0.5
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.5 * -0.5
                                * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 1
                                * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 1 * -0.5
                                * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + 1 * -0.5
                                * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + 1 * 1 * chemtools::compute_dzz_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c20-c21
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        -0.5 * 1.7320508075688772
                            * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 1.7320508075688772
                                * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 1 * 1.7320508075688772
                                * chemtools::compute_dzz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c20-s21
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        -0.5 * 1.7320508075688772
                            * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 1.7320508075688772
                                * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 1 * 1.7320508075688772
                                * chemtools::compute_dzz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c20-c22
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        -0.5 * 0.8660254037844386
                            * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * -0.8660254037844386
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 0.8660254037844386
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.5 * -0.8660254037844386
                                * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 1 * 0.8660254037844386
                                * chemtools::compute_dxx_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + 1 * -0.8660254037844386
                                * chemtools::compute_dyy_dzz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c20-s22
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        -0.5 * 1.7320508075688772
                            * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.5 * 1.7320508075688772
                                * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 1 * 1.7320508075688772
                                * chemtools::compute_dzz_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            i_integral = ((i_row + 1) * knbasis - ((i_row + 1) * ((i_row + 1) - 1)) / 2) + j_col
                - (i_row + 1);// Move 1 row down.
            // Integrate c21-c20
            if (jcontr_shell != icontr_shell) {
              d_point_charge[point_index + (i_integral + 0) * npoints] +=
                  g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                      g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                      chemtools::normalization_primitive_pure_d(alpha) *
                      chemtools::normalization_primitive_pure_d(beta) *
                      (
                          1.7320508075688772 * -0.5
                              * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * -0.5
                                  * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * 1
                                  * chemtools::compute_dzz_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                      );
            }
            // Integrate c21-c21
            d_point_charge[point_index + (i_integral + 1) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dxz_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c21-s21
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dxz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c21-c22
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 0.8660254037844386
                            * chemtools::compute_dxx_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + 1.7320508075688772 * -0.8660254037844386
                                * chemtools::compute_dyy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate c21-s22
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dxy_dxz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 2) * knbasis - ((i_row + 2) * ((i_row + 2) - 1)) / 2) + j_col
                - (i_row + 2);// Move 1 row down.
            // Integrate s21-c20
            if (jcontr_shell != icontr_shell) {
              d_point_charge[point_index + (i_integral + 0) * npoints] +=
                  g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                      g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                      chemtools::normalization_primitive_pure_d(alpha) *
                      chemtools::normalization_primitive_pure_d(beta) *
                      (
                          1.7320508075688772 * -0.5
                              * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * -0.5
                                  * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * 1
                                  * chemtools::compute_dzz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
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
                          1.7320508075688772 * 1.7320508075688772
                              * chemtools::compute_dxz_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                      );
            }
            // Integrate s21-s21
            d_point_charge[point_index + (i_integral + 2) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dyz_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate s21-c22
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 0.8660254037844386
                            * chemtools::compute_dxx_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + 1.7320508075688772 * -0.8660254037844386
                                * chemtools::compute_dyy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            // Integrate s21-s22
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dxy_dyz_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                    );
            i_integral = ((i_row + 3) * knbasis - ((i_row + 3) * ((i_row + 3) - 1)) / 2) + j_col
                - (i_row + 3);// Move 1 row down.
            // Integrate c22-c20
            if (jcontr_shell != icontr_shell) {
              d_point_charge[point_index + (i_integral + 0) * npoints] +=
                  g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                      g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                      chemtools::normalization_primitive_pure_d(alpha) *
                      chemtools::normalization_primitive_pure_d(beta) *
                      (
                          0.8660254037844386 * -0.5
                              * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + 0.8660254037844386 * -0.5
                                  * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + 0.8660254037844386 * 1
                                  * chemtools::compute_dxx_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + -0.8660254037844386 * -0.5
                                  * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + -0.8660254037844386 * -0.5
                                  * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + -0.8660254037844386 * 1
                                  * chemtools::compute_dyy_dzz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
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
                          0.8660254037844386 * 1.7320508075688772
                              * chemtools::compute_dxx_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + -0.8660254037844386 * 1.7320508075688772
                                  * chemtools::compute_dyy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
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
                          0.8660254037844386 * 1.7320508075688772
                              * chemtools::compute_dxx_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                              + -0.8660254037844386 * 1.7320508075688772
                                  * chemtools::compute_dyy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                      );
            }
            // Integrate c22-c22
            d_point_charge[point_index + (i_integral + 3) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        0.8660254037844386 * 0.8660254037844386
                            * chemtools::compute_dxx_dxx_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + 0.8660254037844386 * -0.8660254037844386
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386 * 0.8660254037844386
                                * chemtools::compute_dxx_dyy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                            + -0.8660254037844386 * -0.8660254037844386
                                * chemtools::compute_dyy_dyy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            // Integrate c22-s22
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        0.8660254037844386 * 1.7320508075688772
                            * chemtools::compute_dxx_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                            + -0.8660254037844386 * 1.7320508075688772
                                * chemtools::compute_dyy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            i_integral = ((i_row + 4) * knbasis - ((i_row + 4) * ((i_row + 4) - 1)) / 2) + j_col
                - (i_row + 4);// Move 1 row down.
            // Integrate s22-c20
            if (jcontr_shell != icontr_shell) {
              d_point_charge[point_index + (i_integral + 0) * npoints] +=
                  g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                      g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                      chemtools::normalization_primitive_pure_d(alpha) *
                      chemtools::normalization_primitive_pure_d(beta) *
                      (
                          1.7320508075688772 * -0.5
                              * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * -0.5
                                  * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * 1
                                  * chemtools::compute_dzz_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
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
                          1.7320508075688772 * 1.7320508075688772
                              * chemtools::compute_dxy_dxz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
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
                          1.7320508075688772 * 1.7320508075688772
                              * chemtools::compute_dxy_dyz_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
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
                          1.7320508075688772 * 0.8660254037844386
                              * chemtools::compute_dxx_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                              + 1.7320508075688772 * -0.8660254037844386
                                  * chemtools::compute_dyy_dxy_nuclear_attraction_integral(beta, B, alpha, A, pt, P)
                      );
            }
            // Integrate s22-s22
            d_point_charge[point_index + (i_integral + 4) * npoints] +=
                g_constant_basis[iconst + numb_primitives1 + i_prim1] *
                    g_constant_basis[jconst + numb_primitives2 + i_prim2] *
                    chemtools::normalization_primitive_pure_d(alpha) *
                    chemtools::normalization_primitive_pure_d(beta) *
                    (
                        1.7320508075688772 * 1.7320508075688772
                            * chemtools::compute_dxy_dxy_nuclear_attraction_integral(alpha, A, beta, B, pt, P)
                    );
            break;
        } // End Switch
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
        std::printf("F-type orbitals is depreciated");
        assert(1 == 0);
//        compute_row_f_type_integral(
//            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
//            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
//            );
      }
      else if (angmom_1 == -2) {
        compute_row_dp_type_integral(
            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
            );
      }
      else if (angmom_1 == -3) {
        std::printf("F-type orbitals is depreciated");
        assert(1 == 0);
//        compute_row_fp_type_integral(
//            A, pt, numb_primitives_1, d_point_charge, point_index,  i_row, iconst, jconst,
//            knbasis, knumb_pts, numb_contracted_shells, icontr_shell, screen_tol
//            );
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
    chemtools::IOData& iodata, double* grid, int knumb_pts, const double screen_tol, const bool disp) {
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
  cublasCreate(&handle);

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
  chemtools::cublas_check_errors(cublasSetMatrix (t_nbasis, t_nbasis, sizeof(double), iodata.GetMOOneRDM(),
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
  size_t t_numb_pts_of_each_chunk = free_mem / (sizeof(double) * (t_nbasis * (t_nbasis + 1) + 3));
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
//    printf("Print grid \n");
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_grid);
//    cudaDeviceSynchronize();

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
//    printf("Print d_point charge \n");
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_point_charge);
//    chemtools::print_all<<<1, 1>>>(d_point_charge, t_nbasis * (t_nbasis + 1) / 2);
//    cudaDeviceSynchronize();
    // Free Grid in Device
    cudaFree(d_grid);

    // Transpose point_charge from (Z, Y, X) (col-major) to (Y, X, Z), where Z=number of points, Y, X are the contractions.
    chemtools::cuda_check_errors(cudaMalloc((double **)&d_point_charge_transpose, t_total_size_integrals_ith_iter_bytes));
    chemtools::cublas_check_errors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                            nbasisfuncs * (nbasisfuncs + 1) / 2, (int) t_numb_pts_ith_iter,
                                            &alpha, d_point_charge, (int) t_numb_pts_ith_iter,
                                            &beta, d_point_charge, (int) t_numb_pts_ith_iter,
                                            d_point_charge_transpose, nbasisfuncs * (nbasisfuncs + 1) / 2));
    //printf("Print d point charge transpose \n");
    //print_final_ten_elements<<<1, 1>>>(d_point_charge_transpose, t_total_numb_integrals_ith_iter);
//    chemtools::print_first_ten_elements<<<1, 1>>>(d_point_charge_transpose);
//    cudaDeviceSynchronize();
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
      chemtools::cublas_check_errors(cublasDtpttr(handle, CUBLAS_FILL_MODE_LOWER,
                                               nbasisfuncs,
                                               &d_point_charge_transpose[i * t_nbasis * (t_nbasis + 1) / 2],
                                               d_triangular_format,
                                               nbasisfuncs
      ));
      // Symmetric Matrix multiplication by the identity matrix to convert it to a full-matrix
      //  TODO: THIS PART IS SLOW, IT COULD BE SOMEHOW REMOVED>
      chemtools::cublas_check_errors(cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
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
  cublasDestroy(handle);  // cublas handle is no longer needed infact most of
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
