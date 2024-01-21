#ifndef GBASIS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
#define GBASIS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_

#include "../include/boys_functions.cuh"
#include "../include/cuda_utils.cuh"

#include <stdio.h>

#include "type_traits"

namespace gbasis {
/**
@Section: Recursive E Coefficients from McMurchie-Davidson Equations
--------------------------------------------------------------------
Equation  (9.5.6), (9.5.7) from Helgaker's et al book.
*/
template<int T, int I, int J>
__device__
inline
typename std::enable_if<(T < 0) || (T > (I + J)), double>::type
E(const double &alpha, const double &A_coord, const double &beta, const double &B_coord) {
  return 0.;
}

template<int T, int I, int J>
__device__
inline
typename std::enable_if<(T == 0) && (I == 0) && (J == 0) && (T <= (I + J)), double>::type
E(const double &alpha, const double &A_coord, const double &beta, const double &B_coord) {
  return exp(-(alpha * beta) * pow(A_coord - B_coord, 2) / (alpha + beta));
}

template<int T, int I, int J>
__device__
inline
//typename std::enable_if<(J == 0) && ((I != 0) || (J != 0) || (T != 0)) && ((T >= 0) && (T <= (I + J))), double>::type
typename std::enable_if<(J == 0) &&
                        not((T < 0) || (T > (I + J))) &&
                        not((T == 0) && (I == 0) && (J == 0) && (T <= (I + J))), double>::type
E(const double &alpha, const double &A_coord, const double &beta, const double &B_coord) {
  // Decrement the index i.
  return E<T - 1, I - 1, J>(alpha, A_coord, beta, B_coord) / (2. * (alpha + beta)) -
      E<T, I - 1, J>(alpha, A_coord, beta, B_coord) * (alpha * beta / (alpha + beta)) * (A_coord - B_coord) / alpha +
      (T + 1) * E<T + 1, I - 1, J>(alpha, A_coord, beta, B_coord);
}

template<int T, int I, int J>
__device__
inline
//typename std::enable_if<(J != 0) && ((I != 0) || (J != 0) || (T != 0)) && ((T >= 0) && (T <= (I + J))), double>::type
typename std::enable_if<not (J == 0) &&
                        not((T < 0) || (T > (I + J))) &&
                        not((T == 0) && (I == 0) && (J == 0) && (T <= (I + J))), double>::type
E(const double& alpha, const double& A_coord, const double& beta, const double& B_coord) {
  // Decrement the index j.
  return E<T - 1, I, J - 1>(alpha, A_coord, beta, B_coord) / (2.0 * (alpha + beta)) +
      E<T, I, J - 1>(alpha, A_coord, beta, B_coord) * (alpha * beta / (alpha + beta)) * (A_coord - B_coord) / beta +
      (T + 1) * E<T + 1, I, J - 1>(alpha, A_coord, beta, B_coord);
}

/**
@Section: Recursive R Coefficients from McMurchie-Davidson Equations
--------------------------------------------------------------------
Equation  (9.9.18), (9.9.19), (9.9.20) from Helgaker's et al book.
*/

//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(N == 0) && (T == 0) && (U == 0) && (V == 0), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  return boys0((alpha + beta) * (pow(P_coord.x - C_coord.x, 2.0) +
//                                    pow(P_coord.y - C_coord.y, 2.0) +
//                                    pow(P_coord.z - C_coord.z, 2.0)));
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(N == 1) && (T == 0) && (U == 0) && (V == 0), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  double p = (alpha + beta);
//  return (-2.0 * p) * boys1(p * (pow(P_coord.x - C_coord.x, 2.0) +
//                                     pow(P_coord.y - C_coord.y, 2.0) +
//                                     pow(P_coord.z - C_coord.z, 2.0)));
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(N == 2) && (T == 0) && (U == 0) && (V == 0), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  double p = (alpha + beta);
//  return pow(-2.0 * p, 2) * boys2(p * (pow(P_coord.x - C_coord.x, 2.0) +
//                                      pow(P_coord.y - C_coord.y, 2.0) +
//                                      pow(P_coord.z - C_coord.z, 2.0)));
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(N > 2) && (T == 0) && (U == 0) && (V == 0), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  double p = (alpha + beta);
//  return pow(-2.0 * p, N) * boys(N, p * (pow(P_coord.x - C_coord.x, 2.0) +
//                                                pow(P_coord.y - C_coord.y, 2.0) +
//                                                pow(P_coord.z - C_coord.z, 2.0)));
//}
//
//
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T == 0) && (U == 0) && (V == 1) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  return (P_coord.z - C_coord.z) * R<N + 1, T, U, 0>(alpha, P_coord, beta, C_coord);
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T == 0) && (U == 0) && (V > 1) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  return (V - 1) * R<N + 1, T, U, V - 2>(alpha, P_coord, beta, C_coord) +
//      (P_coord.z - C_coord.z) * R<N + 1, T, U, V - 1>(alpha, P_coord, beta, C_coord);
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T == 0) && (U == 1) &&
//    not ((T == 0) && (U == 0) && (V >= 1)) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
//  return (P_coord.y - C_coord.y) * R<N + 1, T, 0, V>(alpha, P_coord, beta, C_coord);
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T == 0) && (U > 1)&&
//    not ((T == 0) && (U == 0) && (V >= 1)) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3& C_coord) {
//  return (U - 1) * R<N + 1, T, U - 2, V>(alpha, P_coord, beta, C_coord) +
//      (P_coord.y - C_coord.y) * R<N + 1, T, U - 1, V>(alpha, P_coord, beta, C_coord);
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T == 1) &&
//    not ((T == 0) && (U >= 1)) &&
//    not ((T == 0) && (U == 0) && (V >= 1)) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3& C_coord) {
//  return (P_coord.x - C_coord.x) * R<N + 1, 0, U, V>(alpha, P_coord, beta, C_coord);
//}
//
//template<int N, int T, int U, int V>
//__device__
//inline
//typename std::enable_if<(T > 1) &&
//    not ((T == 0) && (U >= 1)) &&
//    not ((T == 0) && (U == 0) && (V >= 1)) &&
//    not ((N >= 0) && (T == 0) && (U == 0) && (V == 0)), double>::type
//R(const double &alpha, const double3 &P_coord, const double &beta, const double3& C_coord) {
//  return (T - 1) * R<N + 1, T - 2, U, V>(alpha, P_coord, beta, C_coord) +
//      (P_coord.x - C_coord.x) * R<N + 1, T - 2, U, V>(alpha, P_coord, beta, C_coord);
//}

/**
 * MICHAELS STUFF
 */
template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U == 0) && (V == 0), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  double p = (alpha + beta);
  return pow(-2.0 * p, N) * boys(N, p * (pow(P_coord.x - C_coord.x, 2.0) +
      pow(P_coord.y - C_coord.y, 2.0) +
      pow(P_coord.z - C_coord.z, 2.0)));
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U == 0) && (V == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.z - C_coord.z) * R<N + 1, T, U, 0>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U == 0) && (V > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (V - 1) * R<N + 1, T, U, V - 2>(alpha, P_coord, beta, C_coord) +
      (P_coord.z - C_coord.z) * R<N + 1, T, U, V - 1>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.y - C_coord.y) * R<N + 1, T, 0, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (U - 1) * R<N + 1, T, U - 2, V>(alpha, P_coord, beta, C_coord) +
      (P_coord.y - C_coord.y) * R<N + 1, T, U - 1, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.x - C_coord.x) * R<N + 1, 0, U, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (T - 1) * R<N + 1, T - 2, U, V>(alpha, P_coord, beta, C_coord) +
      (P_coord.x - C_coord.x) * R<N + 1, T - 1, U, V>(alpha, P_coord, beta, C_coord);
}
/*
@Section Functions for computing the integrals
----------------------------------------------
Generated by python file `generate_recursive_coeffs.py` or something like that
*/
__device__ double compute_s_s_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_px_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_s_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_px_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_px_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_py_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_pz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxx_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dxz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_dyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxx_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gzzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gzzzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gyzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyzzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gyyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gyyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gyyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gyyyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxzzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxyyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxzz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyz_gxxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyz_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyy_gxxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyy_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxyy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxz_gxxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxz_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxz_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxy_gxxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxy_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_gxxxx_gxxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);

} // namespace gbasis

#endif //GBASIS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
