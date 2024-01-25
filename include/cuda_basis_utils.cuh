#ifndef CHEMTOOLS_CUDA_INCLUDE_CUDA_BASIS_UTILS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_CUDA_BASIS_UTILS_CUH_

namespace chemtools {
/// Compute the normalization constant of a single primitive Cartesian Gaussian, S-type only.
__device__ double normalization_primitive_s(double alpha);
/// Compute the normalization constant of a single primitive Cartesian Gaussian, P-type only
__device__ double normalization_primitive_p(double alpha);
/// Compute the normalization constant of a single primitive Cartesian Gaussian, D-type only
__device__ double normalization_primitive_d(double alpha, int nx, int ny, int nz);
/// Compute the normalization constant of a single primitive Cartesian Gaussian, F-type only
__device__ double normalization_primitive_f(double alpha, int nx, int ny, int nz);
__device__ double normalization_primitive_g(double alpha, int nx, int ny, int nz);
/// Compute the normalization constant of a single primitive Pure (Spherical Harmonics) Gaussian, D-type only.
__device__ double normalization_primitive_pure_d(double alpha);
__device__ double normalization_primitive_pure_f(double alpha);
__device__ double normalization_primitive_pure_g(double alpha);
// Compute the solid harmonics
__device__ double solid_harmonic_function_d(int m, double r_Ax, double r_Ay, double r_Az);
__device__ double solid_harmonic_function_f(int m, double r_Ax, double r_Ay, double r_Az);
__device__ double solid_harmonic_function_g(int m, double r_Ax, double r_Ay, double r_Az);
}
#endif //CHEMTOOLS_CUDA_INCLUDE_CUDA_BASIS_UTILS_CUH_
