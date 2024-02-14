#include <cassert>

#include "../include/cuda_basis_utils.cuh"
#include "../include/cuda_utils.cuh"

/// Compute the normalization constant of a single primitive Cartesian Gaussian, S-type only.
__device__ double chemtools::normalization_primitive_s(double alpha) {
  double r = sqrt(2.0 * alpha / CUDART_PI_D);
  return r * sqrt(r);
}

/// Compute the normalization constant of a single primitive Cartesian Gaussian, P-type only
__device__ double chemtools::normalization_primitive_p(double alpha) {
  double x = 2.0 * alpha / CUDART_PI_D;
  return sqrt(x * sqrt(x) * 4.0 * alpha);
}

/// Compute the normalization constant of a single primitive Cartesian Gaussian, D-type only
__device__ double chemtools::normalization_primitive_d(double alpha, int nx, int ny, int nz) {
  // (nx, ny, nz) are the angular components and sum to 2
  if (((nx == 1) & (ny == 1)) | ((nx == 1) & (nz == 1)) | ((ny == 1) & (nz == 1))) {
    double r = sqrt(2.0 * alpha / CUDART_PI_D);
    return r * sqrt(r) * 4.0 * alpha;
  } else {
    double x = 2.0 * alpha / CUDART_PI_D;
    double a = 4.0 * alpha;
    return sqrt(x * sqrt(x) * a * a / 3);
  }
}

/// Compute the normalization constant of a single primitive Cartesian Gaussian, F-type only
__device__ double chemtools::normalization_primitive_f(double alpha, int nx, int ny, int nz) {
  // (nx, ny, nz) are the angular components and sum to 3
  bool cond1 = ((nx == 3) & (ny == 0) & (nz == 0)) |
      ((nx == 0) & (ny == 3) & (nz == 0)) |
      ((nx == 0) & (ny == 0) & (nz == 3));
  double x = 2.0 * alpha / CUDART_PI_D;
  double a = 4.0 * alpha;
  double num = x * sqrt(x) * a * a * a;
  if (cond1) {
    return sqrt(num / 15.0);
  }
  bool cond2 = ((nx == 1) & (ny == 1) & (nz == 1));
  if (cond2) {
    return sqrt(num);
  }
  else {
    return sqrt(num / 3.0);
  }
}

/// Compute the normalization constant of a single primitive Cartesian Gaussian, G-type only
__device__ double chemtools::normalization_primitive_g(double alpha, int nx, int ny, int nz) {
  // (nx, ny, nz) are the angular components and sum to 3
  bool cond1 = ((nx == 4) & (ny == 0) & (nz == 0)) |
      ((nx == 0) & (ny == 4) & (nz == 0)) |
      ((nx == 0) & (ny == 0) & (nz == 4));
  double x = 2.0 * alpha / CUDART_PI_D;
  x = x * sqrt(x);
  double a = 4.0 * alpha;
  double num = x * a * a * a * a;
  if (cond1) {
    return sqrt(num / 105.0);
  }
  cond1 = ((nx == 2) & (ny == 2) & (nz == 0)) |
      ((nx == 2) & (ny == 0) & (nz == 2)) |
      ((nx == 0) & (ny == 2) & (nz == 2));
  if (cond1) {
    return sqrt(num / 9.0);
  }
  cond1 = ((nx == 2) & (ny == 1) & (nz == 1)) |
      ((nx == 1) & (ny == 2) & (nz == 1)) |
      ((nx == 1) & (ny == 1) & (nz == 2));
  if (cond1) {
    return sqrt(num / 3.0);
  }
  else {
    // (3, 1, 1), (1, 3, 1) etc
    return sqrt(num / 15.0);
  }
}

/// Compute the normalization constant of a single primitive Pure (Spherical Harmonics) Gaussian, D-type only.
__device__ double chemtools::normalization_primitive_pure_d(double alpha) {
  // Angular momentum L is 2 in this case.
  double a = 4.0 * alpha;
  double x = 2.0 * alpha / CUDART_PI_D;
  x = x * sqrt(x);
  return sqrt(x * a * a / 3.0);
}

__device__ double chemtools::normalization_primitive_pure_f(double alpha) {
  // Angular momentum is L is 3 in this case
  // Formula is ((2a / pi)^1.5  *  (4 a)^L  / (2L - 1)!! )^0.5
  double a = 4.0 * alpha;
  double x = 2.0 * alpha / CUDART_PI_D;
  x = x * sqrt(x);
  return sqrt(x * a * a * a / 15.0);
}

__device__ double chemtools::normalization_primitive_pure_g(double alpha) {
  // Angular momentum is L is 4 in this case
  // Formula is ((2a / pi)^1.5  *  (4 a)^L  / (2L - 1)!! )^0.5
  double a = 4.0 * alpha;
  double x = 2.0 * alpha / CUDART_PI_D;
  x = x * sqrt(x);
  return sqrt(x * a * a * a * a / 105.0);
}

/// Compute the Pure/Harmonic basis functions for d-type shells.
__device__ double chemtools::solid_harmonic_function_d(int m, double r_Ax, double r_Ay, double r_Az) {
  //  In the literature, e.g. in the book Molecular Electronic-Structure Theory by Helgaker, Jørgensen and Olsen,
  //          negative magnetic quantum numbers for pure functions are usually referring to sine-like functions.
  // so m=-2,-1 are sin functions.
  // These multiply (-1)^m * r^l by sin(or cos) by the associated Legendere polynomial by the norm constants grouped
  //      these norms the l and m are plugged in. In fact it is sqrt(2(2 - |m|)!/(2 + |m|)!).
  // These are obtained from the table in pg 211 of Helgeker's book.
  if (m == -2) {
    return sqrt(3.) * r_Ax * r_Ay;
  }
  else if (m == -1) {
    return sqrt(3.) * r_Ay * r_Az;
  }
  else if (m == 0) {
    return (2 * (r_Az * r_Az) - (r_Ax * r_Ax) - (r_Ay * r_Ay)) / 2.0;
  }
  else if (m == 1) {
    return sqrt(3.) * r_Ax * r_Az;
  }
  else if (m == 2) {
    return sqrt(3.0) * ((r_Ax * r_Ax) -  (r_Ay * r_Ay)) / 2.0;
  }
  else {
    assert(0);
  }
  assert (0);
  return 0.;
}

/// Compute the Pure/Harmonic basis functions for f-type shells.
__device__ double chemtools::solid_harmonic_function_f(int m, double r_Ax, double r_Ay, double r_Az) {
  //  In the literature, e.g. in the book Molecular Electronic-Structure Theory by Helgaker, Jørgensen and Olsen,
  //          negative magnetic quantum numbers for pure functions are usually referring to sine-like functions.
  // so m=-2,-1 are sin functions.
  // These multiply (-1)^m * r^l by sin(or cos) by the associated Legendere polynomial by the norm constants grouped
  //      these norms the l and m are plugged in. In fact it is sqrt(2(2 - |m|)!/(2 + |m|)!).
  // These are obtained from the table in pg 211 of Helgeker's book.

  if (m == -3) {
    return sqrt(2.5) * (3.0 * (r_Ax * r_Ax) -  (r_Ay * r_Ay)) * r_Ay / 2.0;
  }
  else if (m == -2) {
    return sqrt(15.0) * r_Ax * r_Ay * r_Az;
  }
  else if (m == -1) {
    return sqrt(1.5) * (4.0 * (r_Az * r_Az) - ((r_Ax * r_Ax) + (r_Ay * r_Ay))) * r_Ay / 2.0;
  }
  else if (m == 0) {
    return (2.0 * (r_Az * r_Az) - 3.0 * ((r_Ax * r_Ax) + (r_Ay * r_Ay))) * r_Az / 2.0;
  }
  else if (m == 1) {
    return sqrt(1.5) * (4.0 * (r_Az * r_Az) - ((r_Ax * r_Ax) + (r_Ay * r_Ay))) * r_Ax / 2.0;
  }
  else if (m == 2) {
    return sqrt(15.0) * ((r_Ax * r_Ax) -  (r_Ay * r_Ay)) * r_Az / 2.0;
  }
  else if (m == 3) {
    return sqrt(2.5) * ((r_Ax * r_Ax) -  3.0 * (r_Ay * r_Ay)) * r_Ax / 2.0;
  }
  else {
    assert(0);
  }
  assert (0);
  return 0.;
}


/// Compute the Pure/Harmonic basis functions for g-type shells.
__device__ double chemtools::solid_harmonic_function_g(int m, double r_Ax, double r_Ay, double r_Az) {
  if (m == 4) {
    return sqrt(35.0) * ((r_Ax * r_Ax * r_Ax * r_Ax) - 6.0 * (r_Ax * r_Ax) * (r_Ay * r_Ay) + (r_Ay * r_Ay * r_Ay * r_Ay)) / 8.0;
  }
  else if (m == 3) {
    return sqrt(35.0 / 2.0) * ((r_Ax * r_Ax) - 3.0 * (r_Ay * r_Ay)) * r_Ax * r_Az / 2.0;
  }
  else if (m == 2) {
    return sqrt(5.0) * (6.0 * (r_Az * r_Az) -
        ((r_Ay * r_Ay) + (r_Ax * r_Ax)))
        * ((r_Ax * r_Ax) - (r_Ay * r_Ay)) / 4.0;
  }
  else if (m == 1) {
    return sqrt(2.5) * (4.0 * (r_Az * r_Az) - 3.0 * (r_Ax * r_Ax) - 3.0 * (r_Ay * r_Ay))
        * r_Ax * r_Az / 2.0;
  }
  else if (m == 0) {
    double rad_sq = ((r_Az * r_Az) + (r_Ay * r_Ay) + (r_Ax * r_Ax));
    return (
        35.0 * (r_Az * r_Az * r_Az * r_Az) -
            30.0 * (r_Az * r_Az) * rad_sq +
            3.0 * rad_sq * rad_sq
    ) / 8.0;
  }
  else if (m == -1) {
    return sqrt(2.5) * (7.0 * (r_Az * r_Az) -
        3.0 *  ((r_Az * r_Az) + (r_Ay * r_Ay) + (r_Ax * r_Ax)))
        * r_Ay * r_Az / 2.0;
  }
  else if (m == -2) {
    return sqrt(5.0) * (7.0 * (r_Az * r_Az) -
        ((r_Az * r_Az) + (r_Ay * r_Ay) + (r_Ax * r_Ax)))
        * r_Ax * r_Ay / 2.0;
  }
  else if (m == -3) {
    return sqrt(35.0 / 2.0) * (3.0 * (r_Ax * r_Ax) - (r_Ay * r_Ay)) * r_Ay * r_Az / 2.0;
  }
  else if (m == -4) {
    return sqrt(35.0) * ((r_Ax * r_Ax) - (r_Ay * r_Ay)) * r_Ax * r_Ay / 2.0;
  }
  else {
    assert(0);
  }
  assert (0);
  return 0.;
}
