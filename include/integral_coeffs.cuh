#ifndef CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_

#include "../include/boys_functions.cuh"
#include "../include/cuda_utils.cuh"

#include <stdio.h>

#include "type_traits"

namespace chemtools {
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
  return exp(-(alpha * beta) * (A_coord - B_coord) * (A_coord - B_coord) / (alpha + beta));
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

template<int N, int T, int U, int V>
__device__
inline
typename std::enable_if<(T == 0) && (U == 0) && (V == 0), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  double p = (alpha + beta);
  double x = P_coord.x - C_coord.x;
  double y = P_coord.y - C_coord.y;
  double z = P_coord.z - C_coord.z;
  double param = p * (x * x + y * y + z * z);
  return pow(-2.0 * p, __int2double_rn(N)) * chemtools::boys_function<N>(param);
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
__device__ double compute_fyyy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fzzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxxz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fyyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);
__device__ double compute_fxyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P);

} // namespace chemtools

#endif //CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
