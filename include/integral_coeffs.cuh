#ifndef CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_

#include "../include/boys_functions.cuh"
#include "../include/cuda_utils.cuh"

#include <stdio.h>
#include <math.h>
#include <type_traits>

namespace chemtools {
/**
@Section: Recursive E Coefficients from McMurchie-Davidson Equations
--------------------------------------------------------------------
Equation  (9.5.6), (9.5.7) from Helgaker's et al book.
*/
template<int T, int I, int J>
__device__
__forceinline__
typename std::enable_if<(T < 0) || (T > (I + J)), double>::type
E(const double &alpha, const double &A_coord, const double &beta, const double &B_coord) {
  return 0.;
}

template<int T, int I, int J>
__device__
__forceinline__
typename std::enable_if<(T == 0) && (I == 0) && (J == 0) && (T <= (I + J)), double>::type
E(const double &alpha, const double &A_coord, const double &beta, const double &B_coord) {
  return exp(-(alpha * beta) * pow(A_coord - B_coord, 2) / (alpha + beta));
}

template<int T, int I, int J>
__device__
__forceinline__
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
__forceinline__
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
__forceinline__
typename std::enable_if<(T == 0) && (U == 0) && (V == 0), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  double p = (alpha + beta);
  double param = p * (pow(P_coord.x - C_coord.x, 2.0) + pow(P_coord.y - C_coord.y, 2.0) + pow(P_coord.z - C_coord.z, 2.0));
  return pow(-2.0 * p, N) * chemtools::boys_function<N>(param);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T == 0) && (U == 0) && (V == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.z - C_coord.z) * R<N + 1, T, U, 0>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T == 0) && (U == 0) && (V > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (V - 1) * R<N + 1, T, U, V - 2>(alpha, P_coord, beta, C_coord) +
      (P_coord.z - C_coord.z) * R<N + 1, T, U, V - 1>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T == 0) && (U == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.y - C_coord.y) * R<N + 1, T, 0, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T == 0) && (U > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (U - 1) * R<N + 1, T, U - 2, V>(alpha, P_coord, beta, C_coord) +
      (P_coord.y - C_coord.y) * R<N + 1, T, U - 1, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T == 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (P_coord.x - C_coord.x) * R<N + 1, 0, U, V>(alpha, P_coord, beta, C_coord);
}

template<int N, int T, int U, int V>
__device__
__forceinline__
typename std::enable_if<(T > 1), double>::type
R(const double &alpha, const double3 &P_coord, const double &beta, const double3 &C_coord) {
  return (T - 1) * R<N + 1, T - 2, U, V>(alpha, P_coord, beta, C_coord) +
      (P_coord.x - C_coord.x) * R<N + 1, T - 1, U, V>(alpha, P_coord, beta, C_coord);
}


/*
 * Nuclear-Attraction Integrals
 */
__device__ inline double compute_s_s_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_000_zz *  R_000;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_px_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_000_zz *  R_100;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_000_zz *  R_010;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_101_zz *  R_001;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_000_zz *  R_200;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_000_zz *  R_020;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_202_zz *  R_002;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_000_zz *  R_110;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_101_zz *  R_101;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_101_zz *  R_011;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_000_zz *  R_300;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_003_yy *  E_000_zz *  R_000;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_000_zz *  R_010;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_000_zz *  R_020;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_000_zz *  R_030;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_303_zz *  R_003;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_000_zz *  R_120;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_000_zz *  R_210;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_101_zz *  R_201;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_202_zz *  R_102;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_202_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_202_zz *  R_012;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_101_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_101_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_101_zz *  R_021;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_s_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_101_zz *  R_111;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_px_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_000_zz *  R_200;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_000_zz *  R_110;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_101_zz *  R_101;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_000_zz *  R_300;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_000_zz *  R_120;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_202_zz *  R_102;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_000_zz *  R_210;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_101_zz *  R_201;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_101_zz *  R_111;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_013_xx = E<0, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_013_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_113_xx = E<1, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_113_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_213_xx = E<2, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_213_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_313_xx = E<3, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_313_xx *  E_000_yy *  E_000_zz *  R_300;
  double E_413_xx = E<4, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_413_xx *  E_000_yy *  E_000_zz *  R_400;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_003_yy *  E_000_zz *  R_000;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_103_yy *  E_000_zz *  R_010;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_203_yy *  E_000_zz *  R_020;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_303_yy *  E_000_zz *  R_030;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_003_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_103_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_203_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_303_yy *  E_000_zz *  R_130;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_000_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_303_zz *  R_003;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_003_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_103_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_203_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_303_zz *  R_103;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_202_yy *  E_000_zz *  R_120;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_002_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_102_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_202_yy *  E_000_zz *  R_220;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_101_yy *  E_000_zz *  R_210;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_001_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_101_yy *  E_000_zz *  R_310;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_101_zz *  R_201;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_101_zz *  R_301;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_202_zz *  R_102;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_202_zz *  R_202;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_202_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_202_zz *  R_012;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_202_zz *  R_112;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_002_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_101_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_101_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_101_zz *  R_021;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_101_zz *  R_121;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_px_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_101_zz *  R_111;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_101_zz *  R_211;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_py_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_000_zz *  R_020;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_101_zz *  R_011;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_010_yy *  E_000_zz *  R_000;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_000_zz *  R_010;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_000_zz *  R_110;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_000_zz *  R_210;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_012_yy *  E_000_zz *  R_000;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_000_zz *  R_010;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_000_zz *  R_020;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_000_zz *  R_030;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_010_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_202_zz *  R_002;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_202_zz *  R_012;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_000_zz *  R_020;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_000_zz *  R_120;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_101_zz *  R_011;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_101_zz *  R_111;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_101_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_101_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_101_zz *  R_021;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_010_yy *  E_000_zz *  R_000;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_003_xx *  E_110_yy *  E_000_zz *  R_010;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_010_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_110_yy *  E_000_zz *  R_110;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_010_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_110_yy *  E_000_zz *  R_210;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_010_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_110_yy *  E_000_zz *  R_310;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_013_yy = E<0, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_013_yy *  E_000_zz *  R_000;
  double E_113_yy = E<1, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_113_yy *  E_000_zz *  R_010;
  double E_213_yy = E<2, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_213_yy *  E_000_zz *  R_020;
  double E_313_yy = E<3, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_313_yy *  E_000_zz *  R_030;
  double E_413_yy = E<4, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_413_yy *  E_000_zz *  R_040;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_010_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_303_zz *  R_003;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_003_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_103_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_203_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_303_zz *  R_013;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_012_yy *  E_000_zz *  R_000;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_112_yy *  E_000_zz *  R_010;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_212_yy *  E_000_zz *  R_020;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_312_yy *  E_000_zz *  R_030;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_012_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_112_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_212_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_312_yy *  E_000_zz *  R_130;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_211_yy *  E_000_zz *  R_020;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_011_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_111_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_211_yy *  E_000_zz *  R_120;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_011_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_111_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_211_yy *  E_000_zz *  R_220;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_101_zz *  R_011;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_101_zz *  R_111;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_101_zz *  R_211;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_010_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_202_zz *  R_002;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_202_zz *  R_012;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_202_zz *  R_112;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_202_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_202_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_202_zz *  R_022;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_012_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_101_zz *  R_001;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_101_zz *  R_011;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_101_zz *  R_021;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_101_zz *  R_031;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_py_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_011_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_101_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_101_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_101_zz *  R_021;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_101_zz *  R_121;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_pz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_211_zz *  R_002;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_110_zz *  R_001;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_110_zz *  R_101;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_110_zz *  R_201;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_110_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_110_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_110_zz *  R_021;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_312_zz *  R_003;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_110_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_110_zz *  R_011;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_110_zz *  R_111;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_211_zz *  R_002;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_211_zz *  R_102;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_211_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_211_zz *  R_012;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_000_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_003_xx *  E_000_yy *  E_110_zz *  R_001;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_110_zz *  R_101;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_110_zz *  R_201;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_010_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_110_zz *  R_301;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_003_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_003_yy *  E_110_zz *  R_001;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_110_zz *  R_011;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_110_zz *  R_021;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_010_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_110_zz *  R_031;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_013_zz = E<0, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_013_zz *  R_000;
  double E_113_zz = E<1, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_113_zz *  R_001;
  double E_213_zz = E<2, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_213_zz *  R_002;
  double E_313_zz = E<3, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_313_zz *  R_003;
  double E_413_zz = E<4, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_413_zz *  R_004;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_002_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_110_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_110_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_110_zz *  R_021;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_110_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_010_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_110_zz *  R_121;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_001_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_110_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_110_zz *  R_011;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_110_zz *  R_111;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_110_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_010_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_110_zz *  R_211;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_211_zz *  R_002;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_211_zz *  R_102;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_211_zz *  R_202;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_312_zz *  R_003;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_312_zz *  R_103;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_312_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_012_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_112_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_212_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_312_zz *  R_013;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_211_zz *  R_002;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_211_zz *  R_012;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_211_zz *  R_022;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_pz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_211_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_211_zz *  R_012;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_211_zz *  R_112;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_000_zz *  R_300;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_000_zz *  R_400;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_000_zz *  R_120;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_000_zz *  R_220;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_202_zz *  R_102;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_202_zz *  R_202;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_000_zz *  R_210;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_000_zz *  R_310;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_101_zz *  R_201;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_101_zz *  R_301;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_101_zz *  R_111;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_101_zz *  R_211;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_023_xx = E<0, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_023_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_123_xx = E<1, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_123_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_223_xx = E<2, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_223_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_323_xx = E<3, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_323_xx *  E_000_yy *  E_000_zz *  R_300;
  double E_423_xx = E<4, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_423_xx *  E_000_yy *  E_000_zz *  R_400;
  double E_523_xx = E<5, 2, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_500 = R<0, 5, 0, 0>(alpha, P, beta, pt);
  output += E_523_xx *  E_000_yy *  E_000_zz *  R_500;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_003_yy *  E_000_zz *  R_000;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_103_yy *  E_000_zz *  R_010;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_203_yy *  E_000_zz *  R_020;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_303_yy *  E_000_zz *  R_030;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_003_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_103_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_203_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_303_yy *  E_000_zz *  R_130;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_003_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_103_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_203_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_303_yy *  E_000_zz *  R_230;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_000_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_000_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_000_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_020_xx *  E_000_yy *  E_303_zz *  R_003;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_003_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_103_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_203_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_120_xx *  E_000_yy *  E_303_zz *  R_103;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_003_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_103_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_203_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_220_xx *  E_000_yy *  E_303_zz *  R_203;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_202_yy *  E_000_zz *  R_120;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_002_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_102_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_202_yy *  E_000_zz *  R_220;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_002_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_102_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_202_yy *  E_000_zz *  R_320;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_022_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_101_yy *  E_000_zz *  R_210;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_001_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_101_yy *  E_000_zz *  R_310;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_001_yy *  E_000_zz *  R_400;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_101_yy *  E_000_zz *  R_410;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_022_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_101_zz *  R_201;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_101_zz *  R_301;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_001_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_101_zz *  R_401;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_202_zz *  R_102;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_202_zz *  R_202;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_002_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_102_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_202_zz *  R_302;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_001_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_202_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_202_zz *  R_012;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_202_zz *  R_112;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_202_zz *  R_212;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_002_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_002_yy *  E_101_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_101_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_101_zz *  R_021;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_101_zz *  R_121;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_101_zz *  R_221;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxx_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_101_zz *  R_111;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_101_zz *  R_211;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_101_zz *  R_311;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_dyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_022_yy *  E_000_zz *  R_000;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_000_zz *  R_010;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_000_zz *  R_020;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_000_zz *  R_030;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_000_zz *  R_040;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_020_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_020_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_020_yy *  E_202_zz *  R_002;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_202_zz *  R_012;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_202_zz *  R_022;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_021_yy *  E_000_zz *  R_000;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_000_zz *  R_010;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_000_zz *  R_020;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_000_zz *  R_030;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_000_zz *  R_130;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_020_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_020_yy *  E_101_zz *  R_001;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_120_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_120_yy *  E_101_zz *  R_011;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_220_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_220_yy *  E_101_zz *  R_021;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_020_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_020_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_120_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_120_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_220_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_220_yy *  E_101_zz *  R_121;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_021_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_021_yy *  E_101_zz *  R_001;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_121_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_121_yy *  E_101_zz *  R_011;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_221_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_221_yy *  E_101_zz *  R_021;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_321_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_321_yy *  E_101_zz *  R_031;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_020_yy *  E_000_zz *  R_000;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_003_xx *  E_120_yy *  E_000_zz *  R_010;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_003_xx *  E_220_yy *  E_000_zz *  R_020;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_020_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_120_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_220_yy *  E_000_zz *  R_120;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_020_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_120_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_220_yy *  E_000_zz *  R_220;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_020_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_120_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_220_yy *  E_000_zz *  R_320;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_023_yy = E<0, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_023_yy *  E_000_zz *  R_000;
  double E_123_yy = E<1, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_123_yy *  E_000_zz *  R_010;
  double E_223_yy = E<2, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_223_yy *  E_000_zz *  R_020;
  double E_323_yy = E<3, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_323_yy *  E_000_zz *  R_030;
  double E_423_yy = E<4, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_423_yy *  E_000_zz *  R_040;
  double E_523_yy = E<5, 2, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_050 = R<0, 0, 5, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_523_yy *  E_000_zz *  R_050;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_020_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_020_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_020_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_020_yy *  E_303_zz *  R_003;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_003_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_103_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_203_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_120_yy *  E_303_zz *  R_013;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_003_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_103_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_203_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_220_yy *  E_303_zz *  R_023;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_022_yy *  E_000_zz *  R_000;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_122_yy *  E_000_zz *  R_010;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_222_yy *  E_000_zz *  R_020;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_322_yy *  E_000_zz *  R_030;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_422_yy *  E_000_zz *  R_040;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_022_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_122_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_222_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_322_yy *  E_000_zz *  R_130;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_422_yy *  E_000_zz *  R_140;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_021_yy *  E_000_zz *  R_000;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_121_yy *  E_000_zz *  R_010;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_221_yy *  E_000_zz *  R_020;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_321_yy *  E_000_zz *  R_030;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_021_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_121_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_221_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_321_yy *  E_000_zz *  R_130;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_021_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_121_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_221_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_321_yy *  E_000_zz *  R_230;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_020_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_020_yy *  E_101_zz *  R_001;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_120_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_120_yy *  E_101_zz *  R_011;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_220_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_220_yy *  E_101_zz *  R_021;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_020_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_020_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_120_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_120_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_220_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_220_yy *  E_101_zz *  R_121;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_020_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_020_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_120_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_120_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_220_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_220_yy *  E_101_zz *  R_221;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_020_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_020_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_020_yy *  E_202_zz *  R_002;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_120_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_120_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_120_yy *  E_202_zz *  R_012;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_220_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_220_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_220_yy *  E_202_zz *  R_022;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_020_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_020_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_020_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_120_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_120_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_120_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_220_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_220_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_220_yy *  E_202_zz *  R_122;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_021_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_021_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_021_yy *  E_202_zz *  R_002;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_121_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_121_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_121_yy *  E_202_zz *  R_012;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_221_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_221_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_221_yy *  E_202_zz *  R_022;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_321_yy *  E_002_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_321_yy *  E_102_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_321_yy *  E_202_zz *  R_032;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_022_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_022_yy *  E_101_zz *  R_001;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_101_zz *  R_011;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_101_zz *  R_021;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_101_zz *  R_031;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_001_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_101_zz *  R_041;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_021_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_021_yy *  E_101_zz *  R_001;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_101_zz *  R_011;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_101_zz *  R_021;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_101_zz *  R_031;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_101_zz *  R_131;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_dzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_422_zz *  R_004;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_020_zz = E<0, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_020_zz *  R_000;
  double E_120_zz = E<1, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_120_zz *  R_001;
  double E_220_zz = E<2, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_220_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_020_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_120_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_220_zz *  R_012;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_020_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_120_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_220_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_020_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_120_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_220_zz *  R_112;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_321_zz *  R_003;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_321_zz *  R_103;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_321_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_321_zz *  R_013;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_020_zz = E<0, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_000_yy *  E_020_zz *  R_000;
  double E_120_zz = E<1, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_003_xx *  E_000_yy *  E_120_zz *  R_001;
  double E_220_zz = E<2, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_003_xx *  E_000_yy *  E_220_zz *  R_002;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_020_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_120_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_103_xx *  E_000_yy *  E_220_zz *  R_102;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_020_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_120_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_203_xx *  E_000_yy *  E_220_zz *  R_202;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_020_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_120_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_303_xx *  E_000_yy *  E_220_zz *  R_302;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_020_zz = E<0, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_003_yy *  E_020_zz *  R_000;
  double E_120_zz = E<1, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_003_yy *  E_120_zz *  R_001;
  double E_220_zz = E<2, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_003_yy *  E_220_zz *  R_002;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_020_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_120_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_103_yy *  E_220_zz *  R_012;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_020_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_120_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_203_yy *  E_220_zz *  R_022;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_020_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_120_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_303_yy *  E_220_zz *  R_032;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_023_zz = E<0, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_023_zz *  R_000;
  double E_123_zz = E<1, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_123_zz *  R_001;
  double E_223_zz = E<2, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_223_zz *  R_002;
  double E_323_zz = E<3, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_323_zz *  R_003;
  double E_423_zz = E<4, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_423_zz *  R_004;
  double E_523_zz = E<5, 2, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_005 = R<0, 0, 0, 5>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_523_zz *  R_005;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_020_zz = E<0, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_002_yy *  E_020_zz *  R_000;
  double E_120_zz = E<1, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_120_zz *  R_001;
  double E_220_zz = E<2, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_220_zz *  R_002;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_020_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_120_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_220_zz *  R_012;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_020_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_120_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_220_zz *  R_022;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_020_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_120_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_220_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_020_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_120_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_220_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_020_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_120_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_220_zz *  R_122;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_020_zz = E<0, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_001_yy *  E_020_zz *  R_000;
  double E_120_zz = E<1, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_120_zz *  R_001;
  double E_220_zz = E<2, 2, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_220_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_020_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_120_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_220_zz *  R_012;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_020_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_120_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_220_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_020_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_120_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_220_zz *  R_112;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_020_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_120_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_220_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_020_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_120_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_220_zz *  R_212;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_321_zz *  R_003;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_321_zz *  R_103;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_021_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_121_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_221_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_321_zz *  R_203;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_422_zz *  R_004;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_022_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_122_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_222_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_322_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_422_zz *  R_104;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_422_zz *  R_004;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_022_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_122_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_222_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_322_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_422_zz *  R_014;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_321_zz *  R_003;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_321_zz *  R_013;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_021_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_121_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_221_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_321_zz *  R_023;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_321_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_321_zz *  R_013;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_321_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_021_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_121_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_221_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_321_zz *  R_113;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_dxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_000_zz *  R_020;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_000_zz *  R_120;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_000_zz *  R_220;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_110_yy *  E_101_zz *  R_011;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_010_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_010_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_110_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_110_yy *  E_101_zz *  R_111;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_010_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_010_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_110_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_110_yy *  E_101_zz *  R_211;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_011_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_011_yy *  E_101_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_111_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_111_yy *  E_101_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_211_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_211_yy *  E_101_zz *  R_021;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_011_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_011_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_111_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_111_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_211_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_211_yy *  E_101_zz *  R_121;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_013_xx = E<0, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_013_xx *  E_010_yy *  E_000_zz *  R_000;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_013_xx *  E_110_yy *  E_000_zz *  R_010;
  double E_113_xx = E<1, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_113_xx *  E_010_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_113_xx *  E_110_yy *  E_000_zz *  R_110;
  double E_213_xx = E<2, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_213_xx *  E_010_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_213_xx *  E_110_yy *  E_000_zz *  R_210;
  double E_313_xx = E<3, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_313_xx *  E_010_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_313_xx *  E_110_yy *  E_000_zz *  R_310;
  double E_413_xx = E<4, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_413_xx *  E_010_yy *  E_000_zz *  R_400;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_413_xx *  E_110_yy *  E_000_zz *  R_410;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_013_yy = E<0, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_013_yy *  E_000_zz *  R_000;
  double E_113_yy = E<1, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_113_yy *  E_000_zz *  R_010;
  double E_213_yy = E<2, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_213_yy *  E_000_zz *  R_020;
  double E_313_yy = E<3, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_313_yy *  E_000_zz *  R_030;
  double E_413_yy = E<4, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_413_yy *  E_000_zz *  R_040;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_013_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_113_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_213_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_313_yy *  E_000_zz *  R_130;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_413_yy *  E_000_zz *  R_140;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_010_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_010_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_010_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_010_yy *  E_303_zz *  R_003;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_110_yy *  E_003_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_110_yy *  E_103_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_110_yy *  E_203_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_110_yy *  E_303_zz *  R_013;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_010_yy *  E_003_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_010_yy *  E_103_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_010_yy *  E_203_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_010_yy *  E_303_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_110_yy *  E_003_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_110_yy *  E_103_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_110_yy *  E_203_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_110_yy *  E_303_zz *  R_113;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_012_yy *  E_000_zz *  R_000;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_112_yy *  E_000_zz *  R_010;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_212_yy *  E_000_zz *  R_020;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_312_yy *  E_000_zz *  R_030;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_012_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_112_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_212_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_312_yy *  E_000_zz *  R_130;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_012_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_112_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_212_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_312_yy *  E_000_zz *  R_230;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_211_yy *  E_000_zz *  R_020;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_011_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_111_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_211_yy *  E_000_zz *  R_120;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_011_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_111_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_211_yy *  E_000_zz *  R_220;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_011_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_111_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_211_yy *  E_000_zz *  R_320;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_110_yy *  E_101_zz *  R_011;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_010_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_010_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_110_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_110_yy *  E_101_zz *  R_111;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_010_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_010_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_110_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_110_yy *  E_101_zz *  R_211;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_010_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_010_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_110_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_110_yy *  E_101_zz *  R_311;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_010_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_010_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_010_yy *  E_202_zz *  R_002;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_110_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_110_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_110_yy *  E_202_zz *  R_012;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_010_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_010_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_010_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_110_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_110_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_110_yy *  E_202_zz *  R_112;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_010_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_010_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_010_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_110_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_110_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_110_yy *  E_202_zz *  R_212;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_011_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_011_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_011_yy *  E_202_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_111_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_111_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_111_yy *  E_202_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_211_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_211_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_211_yy *  E_202_zz *  R_022;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_011_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_011_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_011_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_111_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_111_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_111_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_211_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_211_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_211_yy *  E_202_zz *  R_122;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_012_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_012_yy *  E_101_zz *  R_001;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_112_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_112_yy *  E_101_zz *  R_011;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_212_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_212_yy *  E_101_zz *  R_021;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_312_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_312_yy *  E_101_zz *  R_031;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_012_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_012_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_112_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_112_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_212_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_212_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_312_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_312_yy *  E_101_zz *  R_131;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_011_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_011_yy *  E_101_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_101_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_101_zz *  R_021;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_101_zz *  R_121;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_101_zz *  R_221;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_dxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_211_zz *  R_002;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_211_zz *  R_102;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_211_zz *  R_202;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_211_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_211_zz *  R_012;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_211_zz *  R_112;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_013_xx = E<0, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_013_xx *  E_000_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_013_xx *  E_000_yy *  E_110_zz *  R_001;
  double E_113_xx = E<1, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_113_xx *  E_000_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_113_xx *  E_000_yy *  E_110_zz *  R_101;
  double E_213_xx = E<2, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_213_xx *  E_000_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_213_xx *  E_000_yy *  E_110_zz *  R_201;
  double E_313_xx = E<3, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_313_xx *  E_000_yy *  E_010_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_313_xx *  E_000_yy *  E_110_zz *  R_301;
  double E_413_xx = E<4, 1, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_413_xx *  E_000_yy *  E_010_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_413_xx *  E_000_yy *  E_110_zz *  R_401;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_003_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_003_yy *  E_110_zz *  R_001;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_103_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_103_yy *  E_110_zz *  R_011;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_203_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_203_yy *  E_110_zz *  R_021;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_303_yy *  E_010_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_303_yy *  E_110_zz *  R_031;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_003_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_003_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_103_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_103_yy *  E_110_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_203_yy *  E_010_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_203_yy *  E_110_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_303_yy *  E_010_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_303_yy *  E_110_zz *  R_131;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_013_zz = E<0, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_000_yy *  E_013_zz *  R_000;
  double E_113_zz = E<1, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_113_zz *  R_001;
  double E_213_zz = E<2, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_213_zz *  R_002;
  double E_313_zz = E<3, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_313_zz *  R_003;
  double E_413_zz = E<4, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_010_xx *  E_000_yy *  E_413_zz *  R_004;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_013_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_113_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_213_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_313_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_110_xx *  E_000_yy *  E_413_zz *  R_104;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_002_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_002_yy *  E_110_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_102_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_102_yy *  E_110_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_202_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_202_yy *  E_110_zz *  R_021;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_002_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_002_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_102_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_102_yy *  E_110_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_202_yy *  E_010_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_202_yy *  E_110_zz *  R_121;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_002_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_002_yy *  E_110_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_102_yy *  E_010_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_102_yy *  E_110_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_202_yy *  E_010_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_202_yy *  E_110_zz *  R_221;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_001_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_001_yy *  E_110_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_101_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_101_yy *  E_110_zz *  R_011;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_001_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_001_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_101_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_101_yy *  E_110_zz *  R_111;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_001_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_001_yy *  E_110_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_101_yy *  E_010_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_101_yy *  E_110_zz *  R_211;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_001_yy *  E_010_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_001_yy *  E_110_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_101_yy *  E_010_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_101_yy *  E_110_zz *  R_311;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_012_xx *  E_000_yy *  E_211_zz *  R_002;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_112_xx *  E_000_yy *  E_211_zz *  R_102;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_212_xx *  E_000_yy *  E_211_zz *  R_202;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_011_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_111_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_312_xx *  E_000_yy *  E_211_zz *  R_302;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_312_zz *  R_003;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_312_zz *  R_103;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_012_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_112_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_212_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_312_zz *  R_203;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_312_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_012_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_112_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_212_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_312_zz *  R_013;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_312_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_012_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_112_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_212_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_312_zz *  R_113;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_002_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_211_zz *  R_002;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_211_zz *  R_012;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_211_zz *  R_022;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_211_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_011_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_111_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_211_zz *  R_122;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dxz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_001_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_211_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_211_zz *  R_012;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_211_zz *  R_112;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_211_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_011_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_111_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_211_zz *  R_212;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_dyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_211_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_211_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_211_zz *  R_022;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_003_xx = E<0, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_003_xx *  E_010_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_003_xx *  E_010_yy *  E_110_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_003_xx *  E_110_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_003_xx *  E_110_yy *  E_110_zz *  R_011;
  double E_103_xx = E<1, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_010_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_103_xx *  E_010_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_103_xx *  E_110_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_103_xx *  E_110_yy *  E_110_zz *  R_111;
  double E_203_xx = E<2, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_010_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_203_xx *  E_010_yy *  E_110_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_203_xx *  E_110_yy *  E_010_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_203_xx *  E_110_yy *  E_110_zz *  R_211;
  double E_303_xx = E<3, 0, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_010_yy *  E_010_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_303_xx *  E_010_yy *  E_110_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_303_xx *  E_110_yy *  E_010_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_303_xx *  E_110_yy *  E_110_zz *  R_311;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_013_yy = E<0, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_013_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_013_yy *  E_110_zz *  R_001;
  double E_113_yy = E<1, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_113_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_113_yy *  E_110_zz *  R_011;
  double E_213_yy = E<2, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_213_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_213_yy *  E_110_zz *  R_021;
  double E_313_yy = E<3, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_313_yy *  E_010_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_313_yy *  E_110_zz *  R_031;
  double E_413_yy = E<4, 1, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_413_yy *  E_010_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_413_yy *  E_110_zz *  R_041;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_013_zz = E<0, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_010_yy *  E_013_zz *  R_000;
  double E_113_zz = E<1, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_113_zz *  R_001;
  double E_213_zz = E<2, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_213_zz *  R_002;
  double E_313_zz = E<3, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_313_zz *  R_003;
  double E_413_zz = E<4, 1, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_010_yy *  E_413_zz *  R_004;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_013_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_113_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_213_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_313_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_110_yy *  E_413_zz *  R_014;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_012_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_012_yy *  E_110_zz *  R_001;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_112_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_112_yy *  E_110_zz *  R_011;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_212_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_212_yy *  E_110_zz *  R_021;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_312_yy *  E_010_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_312_yy *  E_110_zz *  R_031;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_012_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_012_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_112_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_112_yy *  E_110_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_212_yy *  E_010_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_212_yy *  E_110_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_312_yy *  E_010_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_312_yy *  E_110_zz *  R_131;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_010_zz = E<0, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_011_yy *  E_010_zz *  R_000;
  double E_110_zz = E<1, 1, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_011_yy *  E_110_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_111_yy *  E_010_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_111_yy *  E_110_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_211_yy *  E_010_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_211_yy *  E_110_zz *  R_021;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_011_yy *  E_010_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_011_yy *  E_110_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_111_yy *  E_010_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_111_yy *  E_110_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_211_yy *  E_010_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_211_yy *  E_110_zz *  R_121;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_011_yy *  E_010_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_011_yy *  E_110_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_111_yy *  E_010_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_111_yy *  E_110_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_211_yy *  E_010_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_211_yy *  E_110_zz *  R_221;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_010_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_010_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_010_yy *  E_211_zz *  R_002;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_110_yy *  E_211_zz *  R_012;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_010_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_110_yy *  E_211_zz *  R_112;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_010_yy *  E_211_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_011_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_111_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_110_yy *  E_211_zz *  R_212;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_010_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_010_yy *  E_312_zz *  R_003;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_012_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_112_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_212_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_110_yy *  E_312_zz *  R_013;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_010_yy *  E_312_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_012_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_112_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_212_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_110_yy *  E_312_zz *  R_113;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_312_zz *  R_003;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_012_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_112_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_212_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_312_zz *  R_013;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_012_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_112_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_212_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_312_zz *  R_023;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_012_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_211_zz *  R_002;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_211_zz *  R_012;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_211_zz *  R_022;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_011_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_111_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_211_zz *  R_032;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_dyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_011_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_211_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_211_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_211_zz *  R_022;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_211_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_011_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_111_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_211_zz *  R_122;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxxx_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_033_xx = E<0, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_033_xx *  E_000_yy *  E_000_zz *  R_000;
  double E_133_xx = E<1, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_133_xx *  E_000_yy *  E_000_zz *  R_100;
  double E_233_xx = E<2, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_233_xx *  E_000_yy *  E_000_zz *  R_200;
  double E_333_xx = E<3, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_333_xx *  E_000_yy *  E_000_zz *  R_300;
  double E_433_xx = E<4, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_433_xx *  E_000_yy *  E_000_zz *  R_400;
  double E_533_xx = E<5, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_500 = R<0, 5, 0, 0>(alpha, P, beta, pt);
  output += E_533_xx *  E_000_yy *  E_000_zz *  R_500;
  double E_633_xx = E<6, 3, 3>(alpha, A_coord.x, beta, B_coord.x);
  double R_600 = R<0, 6, 0, 0>(alpha, P, beta, pt);
  output += E_633_xx *  E_000_yy *  E_000_zz *  R_600;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_030_xx = E<0, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_003_yy = E<0, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_030_xx *  E_003_yy *  E_000_zz *  R_000;
  double E_103_yy = E<1, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_103_yy *  E_000_zz *  R_010;
  double E_203_yy = E<2, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_203_yy *  E_000_zz *  R_020;
  double E_303_yy = E<3, 0, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_303_yy *  E_000_zz *  R_030;
  double E_130_xx = E<1, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_003_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_103_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_203_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_303_yy *  E_000_zz *  R_130;
  double E_230_xx = E<2, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_003_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_103_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_203_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_303_yy *  E_000_zz *  R_230;
  double E_330_xx = E<3, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_003_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_103_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_203_yy *  E_000_zz *  R_320;
  double R_330 = R<0, 3, 3, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_303_yy *  E_000_zz *  R_330;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_030_xx = E<0, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_030_xx *  E_000_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_000_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_030_xx *  E_000_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_030_xx *  E_000_yy *  E_303_zz *  R_003;
  double E_130_xx = E<1, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_000_yy *  E_003_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_000_yy *  E_103_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_130_xx *  E_000_yy *  E_203_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_130_xx *  E_000_yy *  E_303_zz *  R_103;
  double E_230_xx = E<2, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_000_yy *  E_003_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_000_yy *  E_103_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_230_xx *  E_000_yy *  E_203_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_230_xx *  E_000_yy *  E_303_zz *  R_203;
  double E_330_xx = E<3, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_000_yy *  E_003_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_000_yy *  E_103_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_330_xx *  E_000_yy *  E_203_zz *  R_302;
  double R_303 = R<0, 3, 0, 3>(alpha, P, beta, pt);
  output += E_330_xx *  E_000_yy *  E_303_zz *  R_303;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_031_xx = E<0, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_031_xx *  E_002_yy *  E_000_zz *  R_000;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_031_xx *  E_102_yy *  E_000_zz *  R_010;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_031_xx *  E_202_yy *  E_000_zz *  R_020;
  double E_131_xx = E<1, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_002_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_102_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_202_yy *  E_000_zz *  R_120;
  double E_231_xx = E<2, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_002_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_102_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_202_yy *  E_000_zz *  R_220;
  double E_331_xx = E<3, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_002_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_102_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_202_yy *  E_000_zz *  R_320;
  double E_431_xx = E<4, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_002_yy *  E_000_zz *  R_400;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_102_yy *  E_000_zz *  R_410;
  double R_420 = R<0, 4, 2, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_202_yy *  E_000_zz *  R_420;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_032_xx = E<0, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_032_xx *  E_001_yy *  E_000_zz *  R_000;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_032_xx *  E_101_yy *  E_000_zz *  R_010;
  double E_132_xx = E<1, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_132_xx *  E_001_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_132_xx *  E_101_yy *  E_000_zz *  R_110;
  double E_232_xx = E<2, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_232_xx *  E_001_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_232_xx *  E_101_yy *  E_000_zz *  R_210;
  double E_332_xx = E<3, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_332_xx *  E_001_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_332_xx *  E_101_yy *  E_000_zz *  R_310;
  double E_432_xx = E<4, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_432_xx *  E_001_yy *  E_000_zz *  R_400;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_432_xx *  E_101_yy *  E_000_zz *  R_410;
  double E_532_xx = E<5, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_500 = R<0, 5, 0, 0>(alpha, P, beta, pt);
  output += E_532_xx *  E_001_yy *  E_000_zz *  R_500;
  double R_510 = R<0, 5, 1, 0>(alpha, P, beta, pt);
  output += E_532_xx *  E_101_yy *  E_000_zz *  R_510;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_032_xx = E<0, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_032_xx *  E_000_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_032_xx *  E_000_yy *  E_101_zz *  R_001;
  double E_132_xx = E<1, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_132_xx *  E_000_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_132_xx *  E_000_yy *  E_101_zz *  R_101;
  double E_232_xx = E<2, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_232_xx *  E_000_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_232_xx *  E_000_yy *  E_101_zz *  R_201;
  double E_332_xx = E<3, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_332_xx *  E_000_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_332_xx *  E_000_yy *  E_101_zz *  R_301;
  double E_432_xx = E<4, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_432_xx *  E_000_yy *  E_001_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_432_xx *  E_000_yy *  E_101_zz *  R_401;
  double E_532_xx = E<5, 3, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_500 = R<0, 5, 0, 0>(alpha, P, beta, pt);
  output += E_532_xx *  E_000_yy *  E_001_zz *  R_500;
  double R_501 = R<0, 5, 0, 1>(alpha, P, beta, pt);
  output += E_532_xx *  E_000_yy *  E_101_zz *  R_501;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_031_xx = E<0, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_031_xx *  E_000_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_031_xx *  E_000_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_031_xx *  E_000_yy *  E_202_zz *  R_002;
  double E_131_xx = E<1, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_000_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_131_xx *  E_000_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_131_xx *  E_000_yy *  E_202_zz *  R_102;
  double E_231_xx = E<2, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_000_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_231_xx *  E_000_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_231_xx *  E_000_yy *  E_202_zz *  R_202;
  double E_331_xx = E<3, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_000_yy *  E_002_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_331_xx *  E_000_yy *  E_102_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_331_xx *  E_000_yy *  E_202_zz *  R_302;
  double E_431_xx = E<4, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_000_yy *  E_002_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_431_xx *  E_000_yy *  E_102_zz *  R_401;
  double R_402 = R<0, 4, 0, 2>(alpha, P, beta, pt);
  output += E_431_xx *  E_000_yy *  E_202_zz *  R_402;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_030_xx = E<0, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_030_xx *  E_001_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_001_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_030_xx *  E_001_yy *  E_202_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_101_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_101_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_030_xx *  E_101_yy *  E_202_zz *  R_012;
  double E_130_xx = E<1, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_001_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_001_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_130_xx *  E_001_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_101_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_101_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_130_xx *  E_101_yy *  E_202_zz *  R_112;
  double E_230_xx = E<2, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_001_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_001_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_230_xx *  E_001_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_101_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_101_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_230_xx *  E_101_yy *  E_202_zz *  R_212;
  double E_330_xx = E<3, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_001_yy *  E_002_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_001_yy *  E_102_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_330_xx *  E_001_yy *  E_202_zz *  R_302;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_101_yy *  E_002_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_101_yy *  E_102_zz *  R_311;
  double R_312 = R<0, 3, 1, 2>(alpha, P, beta, pt);
  output += E_330_xx *  E_101_yy *  E_202_zz *  R_312;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_030_xx = E<0, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_030_xx *  E_002_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_002_yy *  E_101_zz *  R_001;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_102_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_102_yy *  E_101_zz *  R_011;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_030_xx *  E_202_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_030_xx *  E_202_yy *  E_101_zz *  R_021;
  double E_130_xx = E<1, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_002_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_002_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_102_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_102_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_130_xx *  E_202_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_130_xx *  E_202_yy *  E_101_zz *  R_121;
  double E_230_xx = E<2, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_002_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_002_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_102_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_102_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_230_xx *  E_202_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_230_xx *  E_202_yy *  E_101_zz *  R_221;
  double E_330_xx = E<3, 3, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_002_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_002_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_102_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_102_yy *  E_101_zz *  R_311;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_330_xx *  E_202_yy *  E_001_zz *  R_320;
  double R_321 = R<0, 3, 2, 1>(alpha, P, beta, pt);
  output += E_330_xx *  E_202_yy *  E_101_zz *  R_321;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxx_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_031_xx = E<0, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_031_xx *  E_001_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_031_xx *  E_001_yy *  E_101_zz *  R_001;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_031_xx *  E_101_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_031_xx *  E_101_yy *  E_101_zz *  R_011;
  double E_131_xx = E<1, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_001_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_131_xx *  E_001_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_131_xx *  E_101_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_131_xx *  E_101_yy *  E_101_zz *  R_111;
  double E_231_xx = E<2, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_001_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_231_xx *  E_001_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_231_xx *  E_101_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_231_xx *  E_101_yy *  E_101_zz *  R_211;
  double E_331_xx = E<3, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_001_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_331_xx *  E_001_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_331_xx *  E_101_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_331_xx *  E_101_yy *  E_101_zz *  R_311;
  double E_431_xx = E<4, 3, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_001_yy *  E_001_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_431_xx *  E_001_yy *  E_101_zz *  R_401;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_431_xx *  E_101_yy *  E_001_zz *  R_410;
  double R_411 = R<0, 4, 1, 1>(alpha, P, beta, pt);
  output += E_431_xx *  E_101_yy *  E_101_zz *  R_411;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fyyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_033_yy = E<0, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_033_yy *  E_000_zz *  R_000;
  double E_133_yy = E<1, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_133_yy *  E_000_zz *  R_010;
  double E_233_yy = E<2, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_233_yy *  E_000_zz *  R_020;
  double E_333_yy = E<3, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_333_yy *  E_000_zz *  R_030;
  double E_433_yy = E<4, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_433_yy *  E_000_zz *  R_040;
  double E_533_yy = E<5, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_050 = R<0, 0, 5, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_533_yy *  E_000_zz *  R_050;
  double E_633_yy = E<6, 3, 3>(alpha, A_coord.y, beta, B_coord.y);
  double R_060 = R<0, 0, 6, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_633_yy *  E_000_zz *  R_060;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_030_yy = E<0, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_003_zz = E<0, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_030_yy *  E_003_zz *  R_000;
  double E_103_zz = E<1, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_030_yy *  E_103_zz *  R_001;
  double E_203_zz = E<2, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_030_yy *  E_203_zz *  R_002;
  double E_303_zz = E<3, 0, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_030_yy *  E_303_zz *  R_003;
  double E_130_yy = E<1, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_130_yy *  E_003_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_130_yy *  E_103_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_130_yy *  E_203_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_130_yy *  E_303_zz *  R_013;
  double E_230_yy = E<2, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_230_yy *  E_003_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_230_yy *  E_103_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_230_yy *  E_203_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_230_yy *  E_303_zz *  R_023;
  double E_330_yy = E<3, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_330_yy *  E_003_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_330_yy *  E_103_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_330_yy *  E_203_zz *  R_032;
  double R_033 = R<0, 0, 3, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_330_yy *  E_303_zz *  R_033;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_032_yy = E<0, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_032_yy *  E_000_zz *  R_000;
  double E_132_yy = E<1, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_132_yy *  E_000_zz *  R_010;
  double E_232_yy = E<2, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_232_yy *  E_000_zz *  R_020;
  double E_332_yy = E<3, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_332_yy *  E_000_zz *  R_030;
  double E_432_yy = E<4, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_432_yy *  E_000_zz *  R_040;
  double E_532_yy = E<5, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_050 = R<0, 0, 5, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_532_yy *  E_000_zz *  R_050;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_032_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_132_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_232_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_332_yy *  E_000_zz *  R_130;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_432_yy *  E_000_zz *  R_140;
  double R_150 = R<0, 1, 5, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_532_yy *  E_000_zz *  R_150;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_031_yy = E<0, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_031_yy *  E_000_zz *  R_000;
  double E_131_yy = E<1, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_131_yy *  E_000_zz *  R_010;
  double E_231_yy = E<2, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_231_yy *  E_000_zz *  R_020;
  double E_331_yy = E<3, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_331_yy *  E_000_zz *  R_030;
  double E_431_yy = E<4, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_431_yy *  E_000_zz *  R_040;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_031_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_131_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_231_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_331_yy *  E_000_zz *  R_130;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_431_yy *  E_000_zz *  R_140;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_031_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_131_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_231_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_331_yy *  E_000_zz *  R_230;
  double R_240 = R<0, 2, 4, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_431_yy *  E_000_zz *  R_240;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_030_yy = E<0, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_030_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_030_yy *  E_101_zz *  R_001;
  double E_130_yy = E<1, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_130_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_130_yy *  E_101_zz *  R_011;
  double E_230_yy = E<2, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_230_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_230_yy *  E_101_zz *  R_021;
  double E_330_yy = E<3, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_330_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_330_yy *  E_101_zz *  R_031;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_030_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_030_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_130_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_130_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_230_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_230_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_330_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_330_yy *  E_101_zz *  R_131;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_030_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_030_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_130_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_130_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_230_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_230_yy *  E_101_zz *  R_221;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_330_yy *  E_001_zz *  R_230;
  double R_231 = R<0, 2, 3, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_330_yy *  E_101_zz *  R_231;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_030_yy = E<0, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_030_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_030_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_030_yy *  E_202_zz *  R_002;
  double E_130_yy = E<1, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_130_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_130_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_130_yy *  E_202_zz *  R_012;
  double E_230_yy = E<2, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_230_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_230_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_230_yy *  E_202_zz *  R_022;
  double E_330_yy = E<3, 3, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_330_yy *  E_002_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_330_yy *  E_102_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_330_yy *  E_202_zz *  R_032;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_030_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_030_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_030_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_130_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_130_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_130_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_230_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_230_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_230_yy *  E_202_zz *  R_122;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_330_yy *  E_002_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_330_yy *  E_102_zz *  R_131;
  double R_132 = R<0, 1, 3, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_330_yy *  E_202_zz *  R_132;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_031_yy = E<0, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_031_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_031_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_031_yy *  E_202_zz *  R_002;
  double E_131_yy = E<1, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_131_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_131_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_131_yy *  E_202_zz *  R_012;
  double E_231_yy = E<2, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_231_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_231_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_231_yy *  E_202_zz *  R_022;
  double E_331_yy = E<3, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_331_yy *  E_002_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_331_yy *  E_102_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_331_yy *  E_202_zz *  R_032;
  double E_431_yy = E<4, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_431_yy *  E_002_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_431_yy *  E_102_zz *  R_041;
  double R_042 = R<0, 0, 4, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_431_yy *  E_202_zz *  R_042;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_032_yy = E<0, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_032_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_032_yy *  E_101_zz *  R_001;
  double E_132_yy = E<1, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_132_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_132_yy *  E_101_zz *  R_011;
  double E_232_yy = E<2, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_232_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_232_yy *  E_101_zz *  R_021;
  double E_332_yy = E<3, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_332_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_332_yy *  E_101_zz *  R_031;
  double E_432_yy = E<4, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_432_yy *  E_001_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_432_yy *  E_101_zz *  R_041;
  double E_532_yy = E<5, 3, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_050 = R<0, 0, 5, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_532_yy *  E_001_zz *  R_050;
  double R_051 = R<0, 0, 5, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_532_yy *  E_101_zz *  R_051;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_031_yy = E<0, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_031_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_031_yy *  E_101_zz *  R_001;
  double E_131_yy = E<1, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_131_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_131_yy *  E_101_zz *  R_011;
  double E_231_yy = E<2, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_231_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_231_yy *  E_101_zz *  R_021;
  double E_331_yy = E<3, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_331_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_331_yy *  E_101_zz *  R_031;
  double E_431_yy = E<4, 3, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_431_yy *  E_001_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_431_yy *  E_101_zz *  R_041;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_031_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_031_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_131_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_131_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_231_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_231_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_331_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_331_yy *  E_101_zz *  R_131;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_431_yy *  E_001_zz *  R_140;
  double R_141 = R<0, 1, 4, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_431_yy *  E_101_zz *  R_141;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fzzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_033_zz = E<0, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_000_yy *  E_033_zz *  R_000;
  double E_133_zz = E<1, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_133_zz *  R_001;
  double E_233_zz = E<2, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_233_zz *  R_002;
  double E_333_zz = E<3, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_333_zz *  R_003;
  double E_433_zz = E<4, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_433_zz *  R_004;
  double E_533_zz = E<5, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_005 = R<0, 0, 0, 5>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_533_zz *  R_005;
  double E_633_zz = E<6, 3, 3>(alpha, A_coord.z, beta, B_coord.z);
  double R_006 = R<0, 0, 0, 6>(alpha, P, beta, pt);
  output += E_000_xx *  E_000_yy *  E_633_zz *  R_006;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_030_zz = E<0, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_002_yy *  E_030_zz *  R_000;
  double E_130_zz = E<1, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_130_zz *  R_001;
  double E_230_zz = E<2, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_230_zz *  R_002;
  double E_330_zz = E<3, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_002_yy *  E_330_zz *  R_003;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_030_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_130_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_230_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_102_yy *  E_330_zz *  R_013;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_030_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_130_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_230_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_202_yy *  E_330_zz *  R_023;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_030_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_130_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_230_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_002_yy *  E_330_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_030_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_130_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_230_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_102_yy *  E_330_zz *  R_113;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_030_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_130_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_230_zz *  R_122;
  double R_123 = R<0, 1, 2, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_202_yy *  E_330_zz *  R_123;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_030_zz = E<0, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_001_yy *  E_030_zz *  R_000;
  double E_130_zz = E<1, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_130_zz *  R_001;
  double E_230_zz = E<2, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_230_zz *  R_002;
  double E_330_zz = E<3, 3, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_002_xx *  E_001_yy *  E_330_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_030_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_130_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_230_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_002_xx *  E_101_yy *  E_330_zz *  R_013;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_030_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_130_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_230_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_102_xx *  E_001_yy *  E_330_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_030_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_130_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_230_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_102_xx *  E_101_yy *  E_330_zz *  R_113;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_030_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_130_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_230_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_202_xx *  E_001_yy *  E_330_zz *  R_203;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_030_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_130_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_230_zz *  R_212;
  double R_213 = R<0, 2, 1, 3>(alpha, P, beta, pt);
  output += E_202_xx *  E_101_yy *  E_330_zz *  R_213;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_002_xx = E<0, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_031_zz = E<0, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_002_xx *  E_000_yy *  E_031_zz *  R_000;
  double E_131_zz = E<1, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_131_zz *  R_001;
  double E_231_zz = E<2, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_231_zz *  R_002;
  double E_331_zz = E<3, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_331_zz *  R_003;
  double E_431_zz = E<4, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_002_xx *  E_000_yy *  E_431_zz *  R_004;
  double E_102_xx = E<1, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_031_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_131_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_231_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_331_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_102_xx *  E_000_yy *  E_431_zz *  R_104;
  double E_202_xx = E<2, 0, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_031_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_131_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_231_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_331_zz *  R_203;
  double R_204 = R<0, 2, 0, 4>(alpha, P, beta, pt);
  output += E_202_xx *  E_000_yy *  E_431_zz *  R_204;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_032_zz = E<0, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_000_yy *  E_032_zz *  R_000;
  double E_132_zz = E<1, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_132_zz *  R_001;
  double E_232_zz = E<2, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_232_zz *  R_002;
  double E_332_zz = E<3, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_332_zz *  R_003;
  double E_432_zz = E<4, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_432_zz *  R_004;
  double E_532_zz = E<5, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_005 = R<0, 0, 0, 5>(alpha, P, beta, pt);
  output += E_001_xx *  E_000_yy *  E_532_zz *  R_005;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_032_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_132_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_232_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_332_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_432_zz *  R_104;
  double R_105 = R<0, 1, 0, 5>(alpha, P, beta, pt);
  output += E_101_xx *  E_000_yy *  E_532_zz *  R_105;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_032_zz = E<0, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_001_yy *  E_032_zz *  R_000;
  double E_132_zz = E<1, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_132_zz *  R_001;
  double E_232_zz = E<2, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_232_zz *  R_002;
  double E_332_zz = E<3, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_332_zz *  R_003;
  double E_432_zz = E<4, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_432_zz *  R_004;
  double E_532_zz = E<5, 3, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_005 = R<0, 0, 0, 5>(alpha, P, beta, pt);
  output += E_000_xx *  E_001_yy *  E_532_zz *  R_005;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_032_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_132_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_232_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_332_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_432_zz *  R_014;
  double R_015 = R<0, 0, 1, 5>(alpha, P, beta, pt);
  output += E_000_xx *  E_101_yy *  E_532_zz *  R_015;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_031_zz = E<0, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_002_yy *  E_031_zz *  R_000;
  double E_131_zz = E<1, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_131_zz *  R_001;
  double E_231_zz = E<2, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_231_zz *  R_002;
  double E_331_zz = E<3, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_331_zz *  R_003;
  double E_431_zz = E<4, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_002_yy *  E_431_zz *  R_004;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_031_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_131_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_231_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_331_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_102_yy *  E_431_zz *  R_014;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_031_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_131_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_231_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_331_zz *  R_023;
  double R_024 = R<0, 0, 2, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_202_yy *  E_431_zz *  R_024;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fzzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_031_zz = E<0, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_001_yy *  E_031_zz *  R_000;
  double E_131_zz = E<1, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_131_zz *  R_001;
  double E_231_zz = E<2, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_231_zz *  R_002;
  double E_331_zz = E<3, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_331_zz *  R_003;
  double E_431_zz = E<4, 3, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_001_xx *  E_001_yy *  E_431_zz *  R_004;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_031_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_131_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_231_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_331_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_001_xx *  E_101_yy *  E_431_zz *  R_014;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_031_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_131_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_231_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_331_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_101_xx *  E_001_yy *  E_431_zz *  R_104;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_031_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_131_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_231_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_331_zz *  R_113;
  double R_114 = R<0, 1, 1, 4>(alpha, P, beta, pt);
  output += E_101_xx *  E_101_yy *  E_431_zz *  R_114;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fxyy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_022_yy *  E_000_zz *  R_000;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_122_yy *  E_000_zz *  R_010;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_222_yy *  E_000_zz *  R_020;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_322_yy *  E_000_zz *  R_030;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_422_yy *  E_000_zz *  R_040;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_022_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_122_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_222_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_322_yy *  E_000_zz *  R_130;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_422_yy *  E_000_zz *  R_140;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_022_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_122_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_222_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_322_yy *  E_000_zz *  R_230;
  double R_240 = R<0, 2, 4, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_422_yy *  E_000_zz *  R_240;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_021_yy *  E_000_zz *  R_000;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_121_yy *  E_000_zz *  R_010;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_221_yy *  E_000_zz *  R_020;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_321_yy *  E_000_zz *  R_030;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_021_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_121_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_221_yy *  E_000_zz *  R_120;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_321_yy *  E_000_zz *  R_130;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_021_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_121_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_221_yy *  E_000_zz *  R_220;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_321_yy *  E_000_zz *  R_230;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_021_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_121_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_221_yy *  E_000_zz *  R_320;
  double R_330 = R<0, 3, 3, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_321_yy *  E_000_zz *  R_330;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_012_xx = E<0, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_012_xx *  E_020_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_020_yy *  E_101_zz *  R_001;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_120_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_120_yy *  E_101_zz *  R_011;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_012_xx *  E_220_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_012_xx *  E_220_yy *  E_101_zz *  R_021;
  double E_112_xx = E<1, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_020_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_020_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_120_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_120_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_112_xx *  E_220_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_112_xx *  E_220_yy *  E_101_zz *  R_121;
  double E_212_xx = E<2, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_020_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_020_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_120_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_120_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_212_xx *  E_220_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_212_xx *  E_220_yy *  E_101_zz *  R_221;
  double E_312_xx = E<3, 1, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_020_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_020_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_120_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_120_yy *  E_101_zz *  R_311;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_312_xx *  E_220_yy *  E_001_zz *  R_320;
  double R_321 = R<0, 3, 2, 1>(alpha, P, beta, pt);
  output += E_312_xx *  E_220_yy *  E_101_zz *  R_321;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_020_yy = E<0, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_020_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_020_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_020_yy *  E_202_zz *  R_002;
  double E_120_yy = E<1, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_120_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_120_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_120_yy *  E_202_zz *  R_012;
  double E_220_yy = E<2, 2, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_220_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_220_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_220_yy *  E_202_zz *  R_022;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_020_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_020_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_020_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_120_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_120_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_120_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_220_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_220_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_220_yy *  E_202_zz *  R_122;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_020_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_020_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_020_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_120_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_120_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_120_yy *  E_202_zz *  R_212;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_220_yy *  E_002_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_220_yy *  E_102_zz *  R_221;
  double R_222 = R<0, 2, 2, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_220_yy *  E_202_zz *  R_222;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_021_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_021_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_021_yy *  E_202_zz *  R_002;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_121_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_121_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_121_yy *  E_202_zz *  R_012;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_221_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_221_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_221_yy *  E_202_zz *  R_022;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_321_yy *  E_002_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_321_yy *  E_102_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_321_yy *  E_202_zz *  R_032;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_021_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_021_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_021_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_121_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_121_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_121_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_221_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_221_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_221_yy *  E_202_zz *  R_122;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_321_yy *  E_002_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_321_yy *  E_102_zz *  R_131;
  double R_132 = R<0, 1, 3, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_321_yy *  E_202_zz *  R_132;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_022_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_022_yy *  E_101_zz *  R_001;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_122_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_122_yy *  E_101_zz *  R_011;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_222_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_222_yy *  E_101_zz *  R_021;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_322_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_322_yy *  E_101_zz *  R_031;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_422_yy *  E_001_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_422_yy *  E_101_zz *  R_041;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_022_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_022_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_122_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_122_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_222_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_222_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_322_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_322_yy *  E_101_zz *  R_131;
  double R_140 = R<0, 1, 4, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_422_yy *  E_001_zz *  R_140;
  double R_141 = R<0, 1, 4, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_422_yy *  E_101_zz *  R_141;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_021_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_021_yy *  E_101_zz *  R_001;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_121_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_121_yy *  E_101_zz *  R_011;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_221_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_221_yy *  E_101_zz *  R_021;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_321_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_321_yy *  E_101_zz *  R_031;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_021_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_021_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_121_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_121_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_221_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_221_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_321_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_321_yy *  E_101_zz *  R_131;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_021_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_021_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_121_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_121_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_221_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_221_yy *  E_101_zz *  R_221;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_321_yy *  E_001_zz *  R_230;
  double R_231 = R<0, 2, 3, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_321_yy *  E_101_zz *  R_231;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fxxy_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_000_zz = E<0, 0, 0>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_011_yy *  E_000_zz *  R_000;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_022_xx *  E_111_yy *  E_000_zz *  R_010;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_022_xx *  E_211_yy *  E_000_zz *  R_020;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_011_yy *  E_000_zz *  R_100;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_111_yy *  E_000_zz *  R_110;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_211_yy *  E_000_zz *  R_120;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_011_yy *  E_000_zz *  R_200;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_111_yy *  E_000_zz *  R_210;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_211_yy *  E_000_zz *  R_220;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_011_yy *  E_000_zz *  R_300;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_111_yy *  E_000_zz *  R_310;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_211_yy *  E_000_zz *  R_320;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_011_yy *  E_000_zz *  R_400;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_111_yy *  E_000_zz *  R_410;
  double R_420 = R<0, 4, 2, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_211_yy *  E_000_zz *  R_420;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_010_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_022_xx *  E_010_yy *  E_101_zz *  R_001;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_022_xx *  E_110_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_022_xx *  E_110_yy *  E_101_zz *  R_011;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_010_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_122_xx *  E_010_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_110_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_122_xx *  E_110_yy *  E_101_zz *  R_111;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_010_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_222_xx *  E_010_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_110_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_222_xx *  E_110_yy *  E_101_zz *  R_211;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_010_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_322_xx *  E_010_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_110_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_322_xx *  E_110_yy *  E_101_zz *  R_311;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_010_yy *  E_001_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_422_xx *  E_010_yy *  E_101_zz *  R_401;
  double R_410 = R<0, 4, 1, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_110_yy *  E_001_zz *  R_410;
  double R_411 = R<0, 4, 1, 1>(alpha, P, beta, pt);
  output += E_422_xx *  E_110_yy *  E_101_zz *  R_411;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_010_yy = E<0, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_010_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_010_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_010_yy *  E_202_zz *  R_002;
  double E_110_yy = E<1, 1, 0>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_110_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_110_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_110_yy *  E_202_zz *  R_012;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_010_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_010_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_010_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_110_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_110_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_110_yy *  E_202_zz *  R_112;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_010_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_010_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_010_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_110_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_110_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_110_yy *  E_202_zz *  R_212;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_010_yy *  E_002_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_010_yy *  E_102_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_010_yy *  E_202_zz *  R_302;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_110_yy *  E_002_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_110_yy *  E_102_zz *  R_311;
  double R_312 = R<0, 3, 1, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_110_yy *  E_202_zz *  R_312;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_002_zz = E<0, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_011_yy *  E_002_zz *  R_000;
  double E_102_zz = E<1, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_011_yy *  E_102_zz *  R_001;
  double E_202_zz = E<2, 0, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_011_yy *  E_202_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_111_yy *  E_002_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_111_yy *  E_102_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_111_yy *  E_202_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_211_yy *  E_002_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_211_yy *  E_102_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_211_yy *  E_202_zz *  R_022;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_011_yy *  E_002_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_011_yy *  E_102_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_011_yy *  E_202_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_111_yy *  E_002_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_111_yy *  E_102_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_111_yy *  E_202_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_211_yy *  E_002_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_211_yy *  E_102_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_211_yy *  E_202_zz *  R_122;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_011_yy *  E_002_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_011_yy *  E_102_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_011_yy *  E_202_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_111_yy *  E_002_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_111_yy *  E_102_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_111_yy *  E_202_zz *  R_212;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_211_yy *  E_002_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_211_yy *  E_102_zz *  R_221;
  double R_222 = R<0, 2, 2, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_211_yy *  E_202_zz *  R_222;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_012_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_012_yy *  E_101_zz *  R_001;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_112_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_112_yy *  E_101_zz *  R_011;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_212_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_212_yy *  E_101_zz *  R_021;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_312_yy *  E_001_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_312_yy *  E_101_zz *  R_031;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_012_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_012_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_112_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_112_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_212_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_212_yy *  E_101_zz *  R_121;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_312_yy *  E_001_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_312_yy *  E_101_zz *  R_131;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_012_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_012_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_112_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_112_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_212_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_212_yy *  E_101_zz *  R_221;
  double R_230 = R<0, 2, 3, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_312_yy *  E_001_zz *  R_230;
  double R_231 = R<0, 2, 3, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_312_yy *  E_101_zz *  R_231;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxy_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_001_zz = E<0, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_011_yy *  E_001_zz *  R_000;
  double E_101_zz = E<1, 0, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_011_yy *  E_101_zz *  R_001;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_111_yy *  E_001_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_111_yy *  E_101_zz *  R_011;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_211_yy *  E_001_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_211_yy *  E_101_zz *  R_021;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_011_yy *  E_001_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_011_yy *  E_101_zz *  R_101;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_111_yy *  E_001_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_111_yy *  E_101_zz *  R_111;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_211_yy *  E_001_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_211_yy *  E_101_zz *  R_121;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_011_yy *  E_001_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_011_yy *  E_101_zz *  R_201;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_111_yy *  E_001_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_111_yy *  E_101_zz *  R_211;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_211_yy *  E_001_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_211_yy *  E_101_zz *  R_221;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_011_yy *  E_001_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_011_yy *  E_101_zz *  R_301;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_111_yy *  E_001_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_111_yy *  E_101_zz *  R_311;
  double R_320 = R<0, 3, 2, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_211_yy *  E_001_zz *  R_320;
  double R_321 = R<0, 3, 2, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_211_yy *  E_101_zz *  R_321;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxz_fxxz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_022_xx = E<0, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_022_xx *  E_000_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_022_xx *  E_000_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_022_xx *  E_000_yy *  E_211_zz *  R_002;
  double E_122_xx = E<1, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_122_xx *  E_000_yy *  E_211_zz *  R_102;
  double E_222_xx = E<2, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_222_xx *  E_000_yy *  E_211_zz *  R_202;
  double E_322_xx = E<3, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_011_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_111_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_322_xx *  E_000_yy *  E_211_zz *  R_302;
  double E_422_xx = E<4, 2, 2>(alpha, A_coord.x, beta, B_coord.x);
  double R_400 = R<0, 4, 0, 0>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_011_zz *  R_400;
  double R_401 = R<0, 4, 0, 1>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_111_zz *  R_401;
  double R_402 = R<0, 4, 0, 2>(alpha, P, beta, pt);
  output += E_422_xx *  E_000_yy *  E_211_zz *  R_402;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_000_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_021_xx *  E_000_yy *  E_312_zz *  R_003;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_121_xx *  E_000_yy *  E_312_zz *  R_103;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_012_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_112_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_212_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_221_xx *  E_000_yy *  E_312_zz *  R_203;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_012_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_112_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_212_zz *  R_302;
  double R_303 = R<0, 3, 0, 3>(alpha, P, beta, pt);
  output += E_321_xx *  E_000_yy *  E_312_zz *  R_303;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_012_zz = E<0, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_001_yy *  E_012_zz *  R_000;
  double E_112_zz = E<1, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_112_zz *  R_001;
  double E_212_zz = E<2, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_212_zz *  R_002;
  double E_312_zz = E<3, 1, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_020_xx *  E_001_yy *  E_312_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_012_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_112_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_212_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_020_xx *  E_101_yy *  E_312_zz *  R_013;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_012_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_112_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_212_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_120_xx *  E_001_yy *  E_312_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_012_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_112_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_212_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_120_xx *  E_101_yy *  E_312_zz *  R_113;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_012_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_112_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_212_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_220_xx *  E_001_yy *  E_312_zz *  R_203;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_012_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_112_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_212_zz *  R_212;
  double R_213 = R<0, 2, 1, 3>(alpha, P, beta, pt);
  output += E_220_xx *  E_101_yy *  E_312_zz *  R_213;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_020_xx = E<0, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_020_xx *  E_002_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_002_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_002_yy *  E_211_zz *  R_002;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_102_yy *  E_211_zz *  R_012;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_020_xx *  E_202_yy *  E_211_zz *  R_022;
  double E_120_xx = E<1, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_002_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_102_yy *  E_211_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_011_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_111_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_120_xx *  E_202_yy *  E_211_zz *  R_122;
  double E_220_xx = E<2, 2, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_002_yy *  E_211_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_011_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_111_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_102_yy *  E_211_zz *  R_212;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_011_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_111_zz *  R_221;
  double R_222 = R<0, 2, 2, 2>(alpha, P, beta, pt);
  output += E_220_xx *  E_202_yy *  E_211_zz *  R_222;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxxz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_021_xx = E<0, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_021_xx *  E_001_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_001_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_001_yy *  E_211_zz *  R_002;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_021_xx *  E_101_yy *  E_211_zz *  R_012;
  double E_121_xx = E<1, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_001_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_121_xx *  E_101_yy *  E_211_zz *  R_112;
  double E_221_xx = E<2, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_001_yy *  E_211_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_011_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_111_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_221_xx *  E_101_yy *  E_211_zz *  R_212;
  double E_321_xx = E<3, 2, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_300 = R<0, 3, 0, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_011_zz *  R_300;
  double R_301 = R<0, 3, 0, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_111_zz *  R_301;
  double R_302 = R<0, 3, 0, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_001_yy *  E_211_zz *  R_302;
  double R_310 = R<0, 3, 1, 0>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_011_zz *  R_310;
  double R_311 = R<0, 3, 1, 1>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_111_zz *  R_311;
  double R_312 = R<0, 3, 1, 2>(alpha, P, beta, pt);
  output += E_321_xx *  E_101_yy *  E_211_zz *  R_312;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxzz_fxzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_000_yy = E<0, 0, 0>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_000_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_011_xx *  E_000_yy *  E_422_zz *  R_004;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_022_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_122_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_222_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_322_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_111_xx *  E_000_yy *  E_422_zz *  R_104;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_022_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_122_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_222_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_322_zz *  R_203;
  double R_204 = R<0, 2, 0, 4>(alpha, P, beta, pt);
  output += E_211_xx *  E_000_yy *  E_422_zz *  R_204;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_001_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_010_xx *  E_001_yy *  E_422_zz *  R_004;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_022_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_122_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_222_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_322_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_010_xx *  E_101_yy *  E_422_zz *  R_014;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_022_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_122_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_222_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_322_zz *  R_103;
  double R_104 = R<0, 1, 0, 4>(alpha, P, beta, pt);
  output += E_110_xx *  E_001_yy *  E_422_zz *  R_104;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_022_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_122_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_222_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_322_zz *  R_113;
  double R_114 = R<0, 1, 1, 4>(alpha, P, beta, pt);
  output += E_110_xx *  E_101_yy *  E_422_zz *  R_114;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_010_xx = E<0, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_002_yy = E<0, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_010_xx *  E_002_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_002_yy *  E_321_zz *  R_003;
  double E_102_yy = E<1, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_102_yy *  E_321_zz *  R_013;
  double E_202_yy = E<2, 0, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_021_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_121_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_221_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_010_xx *  E_202_yy *  E_321_zz *  R_023;
  double E_110_xx = E<1, 1, 0>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_002_yy *  E_321_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_021_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_121_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_221_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_102_yy *  E_321_zz *  R_113;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_021_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_121_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_221_zz *  R_122;
  double R_123 = R<0, 1, 2, 3>(alpha, P, beta, pt);
  output += E_110_xx *  E_202_yy *  E_321_zz *  R_123;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_001_yy = E<0, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_001_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_011_xx *  E_001_yy *  E_321_zz *  R_003;
  double E_101_yy = E<1, 0, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_011_xx *  E_101_yy *  E_321_zz *  R_013;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_111_xx *  E_001_yy *  E_321_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_021_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_121_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_221_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_111_xx *  E_101_yy *  E_321_zz *  R_113;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_021_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_121_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_221_zz *  R_202;
  double R_203 = R<0, 2, 0, 3>(alpha, P, beta, pt);
  output += E_211_xx *  E_001_yy *  E_321_zz *  R_203;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_021_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_121_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_221_zz *  R_212;
  double R_213 = R<0, 2, 1, 3>(alpha, P, beta, pt);
  output += E_211_xx *  E_101_yy *  E_321_zz *  R_213;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyzz_fyzz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_022_zz = E<0, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_011_yy *  E_022_zz *  R_000;
  double E_122_zz = E<1, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_122_zz *  R_001;
  double E_222_zz = E<2, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_222_zz *  R_002;
  double E_322_zz = E<3, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_322_zz *  R_003;
  double E_422_zz = E<4, 2, 2>(alpha, A_coord.z, beta, B_coord.z);
  double R_004 = R<0, 0, 0, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_011_yy *  E_422_zz *  R_004;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_022_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_122_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_222_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_322_zz *  R_013;
  double R_014 = R<0, 0, 1, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_111_yy *  E_422_zz *  R_014;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_022_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_122_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_222_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_322_zz *  R_023;
  double R_024 = R<0, 0, 2, 4>(alpha, P, beta, pt);
  output += E_000_xx *  E_211_yy *  E_422_zz *  R_024;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyzz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_012_yy = E<0, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_012_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_012_yy *  E_321_zz *  R_003;
  double E_112_yy = E<1, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_112_yy *  E_321_zz *  R_013;
  double E_212_yy = E<2, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_021_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_121_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_221_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_212_yy *  E_321_zz *  R_023;
  double E_312_yy = E<3, 1, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_021_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_121_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_221_zz *  R_032;
  double R_033 = R<0, 0, 3, 3>(alpha, P, beta, pt);
  output += E_000_xx *  E_312_yy *  E_321_zz *  R_033;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyzz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_021_zz = E<0, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_011_yy *  E_021_zz *  R_000;
  double E_121_zz = E<1, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_121_zz *  R_001;
  double E_221_zz = E<2, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_221_zz *  R_002;
  double E_321_zz = E<3, 2, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_003 = R<0, 0, 0, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_011_yy *  E_321_zz *  R_003;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_021_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_121_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_221_zz *  R_012;
  double R_013 = R<0, 0, 1, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_111_yy *  E_321_zz *  R_013;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_021_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_121_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_221_zz *  R_022;
  double R_023 = R<0, 0, 2, 3>(alpha, P, beta, pt);
  output += E_001_xx *  E_211_yy *  E_321_zz *  R_023;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_021_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_121_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_221_zz *  R_102;
  double R_103 = R<0, 1, 0, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_011_yy *  E_321_zz *  R_103;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_021_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_121_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_221_zz *  R_112;
  double R_113 = R<0, 1, 1, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_111_yy *  E_321_zz *  R_113;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_021_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_121_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_221_zz *  R_122;
  double R_123 = R<0, 1, 2, 3>(alpha, P, beta, pt);
  output += E_101_xx *  E_211_yy *  E_321_zz *  R_123;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyz_fyyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_000_xx = E<0, 0, 0>(alpha, A_coord.x, beta, B_coord.x);
  double E_022_yy = E<0, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_000_xx *  E_022_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_022_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_022_yy *  E_211_zz *  R_002;
  double E_122_yy = E<1, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_122_yy *  E_211_zz *  R_012;
  double E_222_yy = E<2, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_222_yy *  E_211_zz *  R_022;
  double E_322_yy = E<3, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_011_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_111_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_322_yy *  E_211_zz *  R_032;
  double E_422_yy = E<4, 2, 2>(alpha, A_coord.y, beta, B_coord.y);
  double R_040 = R<0, 0, 4, 0>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_011_zz *  R_040;
  double R_041 = R<0, 0, 4, 1>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_111_zz *  R_041;
  double R_042 = R<0, 0, 4, 2>(alpha, P, beta, pt);
  output += E_000_xx *  E_422_yy *  E_211_zz *  R_042;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fyyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_001_xx = E<0, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_021_yy = E<0, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_001_xx *  E_021_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_021_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_021_yy *  E_211_zz *  R_002;
  double E_121_yy = E<1, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_121_yy *  E_211_zz *  R_012;
  double E_221_yy = E<2, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_221_yy *  E_211_zz *  R_022;
  double E_321_yy = E<3, 2, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_030 = R<0, 0, 3, 0>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_011_zz *  R_030;
  double R_031 = R<0, 0, 3, 1>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_111_zz *  R_031;
  double R_032 = R<0, 0, 3, 2>(alpha, P, beta, pt);
  output += E_001_xx *  E_321_yy *  E_211_zz *  R_032;
  double E_101_xx = E<1, 0, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_021_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_121_yy *  E_211_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_011_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_111_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_221_yy *  E_211_zz *  R_122;
  double R_130 = R<0, 1, 3, 0>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_011_zz *  R_130;
  double R_131 = R<0, 1, 3, 1>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_111_zz *  R_131;
  double R_132 = R<0, 1, 3, 2>(alpha, P, beta, pt);
  output += E_101_xx *  E_321_yy *  E_211_zz *  R_132;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
__device__ inline double compute_fxyz_fxyz_nuclear_attraction_integral(const double& alpha, const double3& A_coord, const double& beta, const double3& B_coord, const double3& pt, const double3& P)
{
  double E_011_xx = E<0, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double E_011_yy = E<0, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double E_011_zz = E<0, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_000 = R<0, 0, 0, 0>(alpha, P, beta, pt);
  double output = E_011_xx *  E_011_yy *  E_011_zz *  R_000;
  double E_111_zz = E<1, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_001 = R<0, 0, 0, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_011_yy *  E_111_zz *  R_001;
  double E_211_zz = E<2, 1, 1>(alpha, A_coord.z, beta, B_coord.z);
  double R_002 = R<0, 0, 0, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_011_yy *  E_211_zz *  R_002;
  double E_111_yy = E<1, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_010 = R<0, 0, 1, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_011_zz *  R_010;
  double R_011 = R<0, 0, 1, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_111_zz *  R_011;
  double R_012 = R<0, 0, 1, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_111_yy *  E_211_zz *  R_012;
  double E_211_yy = E<2, 1, 1>(alpha, A_coord.y, beta, B_coord.y);
  double R_020 = R<0, 0, 2, 0>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_011_zz *  R_020;
  double R_021 = R<0, 0, 2, 1>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_111_zz *  R_021;
  double R_022 = R<0, 0, 2, 2>(alpha, P, beta, pt);
  output += E_011_xx *  E_211_yy *  E_211_zz *  R_022;
  double E_111_xx = E<1, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_100 = R<0, 1, 0, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_011_zz *  R_100;
  double R_101 = R<0, 1, 0, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_111_zz *  R_101;
  double R_102 = R<0, 1, 0, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_011_yy *  E_211_zz *  R_102;
  double R_110 = R<0, 1, 1, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_011_zz *  R_110;
  double R_111 = R<0, 1, 1, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_111_zz *  R_111;
  double R_112 = R<0, 1, 1, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_111_yy *  E_211_zz *  R_112;
  double R_120 = R<0, 1, 2, 0>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_011_zz *  R_120;
  double R_121 = R<0, 1, 2, 1>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_111_zz *  R_121;
  double R_122 = R<0, 1, 2, 2>(alpha, P, beta, pt);
  output += E_111_xx *  E_211_yy *  E_211_zz *  R_122;
  double E_211_xx = E<2, 1, 1>(alpha, A_coord.x, beta, B_coord.x);
  double R_200 = R<0, 2, 0, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_011_zz *  R_200;
  double R_201 = R<0, 2, 0, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_111_zz *  R_201;
  double R_202 = R<0, 2, 0, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_011_yy *  E_211_zz *  R_202;
  double R_210 = R<0, 2, 1, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_011_zz *  R_210;
  double R_211 = R<0, 2, 1, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_111_zz *  R_211;
  double R_212 = R<0, 2, 1, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_111_yy *  E_211_zz *  R_212;
  double R_220 = R<0, 2, 2, 0>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_011_zz *  R_220;
  double R_221 = R<0, 2, 2, 1>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_111_zz *  R_221;
  double R_222 = R<0, 2, 2, 2>(alpha, P, beta, pt);
  output += E_211_xx *  E_211_yy *  E_211_zz *  R_222;
  return output * (2.0 * CUDART_PI_D) / (alpha + beta);
}
} // namespace chemtools

#endif //CHEMTOOLS_CUDA_INCLUDE_INTEGRAL_COEFFS_CUH_
