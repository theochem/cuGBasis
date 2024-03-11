#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_GRADIENT_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_GRADIENT_CUH_

#include "cublas_v2.h"

#include "contracted_shell.h"
#include "iodata.h"

using d_func_t = void (*)(double*, const double*, const int, const int, const int);

namespace chemtools {

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
// This points to the correct __device__ function that evaluates over contractions
__device__ extern d_func_t p_evaluate_deriv_contractions;

/**
 * Evaluate the derivative of contractions from any grid of points.
 *
 * Assumes basis set information is stored in GPU constant memory via basis_to_gpu.cu file.
 *
 * The calculation of the derivative is given as follows:
 * Let G_{ijk}(r, a, A) = x_A^i  y_A^j z_A^k e^{-a r_A^2} be a Cartesian primitive Gaussian.
 * Then we have that this factorizes into G_{ijk}(r, a, A) = G_i(a, x_A) G_j(a, y_B) G_k(a, z_A)
 * and that the derivative of this is \frac{\partial G_i}{\partial x} = i G_{i - 1}(a, x_A) - 2 a G_{i+1}(a, x_A)
 *
 * @param[in,out] g_d_constant_basis Pointer to the the output.
 * @param[out] d_deriv_contracs  The device pointer to the contractions array of size (3, M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order, i.e.
 *                  Derivatives first, then contractions then points.
 *                  Values should be set to zero before using.
 * @param[in] d_points The device pointer to the grid points of size (N, 3) stored in column-major order.
 * @param[in] d_knumb_points The number of points in the grid.
 * @param[in] knumb_contractions Total number of contractions, note this may be different on whats in constant memory
 * @param[in] i_contr_start Index of where to start updating contractions, should match whats in constant memory.
 */
__device__ void evaluate_derivatives_contractions_from_constant_memory(
    double* d_deriv_contracs, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_contr_start = 0
);


//__device__ static d_func_t p_evaluate_deriv_contractions;
//__device__ static d_func_t p_evaluate_contractions;

/**
 * HOST FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Evaluate derivatives of the contractions.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
 */
__host__ std::vector<double> evaluate_contraction_derivatives(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Evaluate the gradient of the electron density from molecular basis in constant memory.
 *
 * Note that the electron density is defined as:
 * \rho(r) = \sum w_i \psi_i^\star(r)   \psi_i(r) =
 * = \sum_i \sum_{k, l} w_i c_{k, i}^\star c_{j, i} \phi_k^\star(r) \phi_l(r),
 *
 * where w_i is the occupany and \psi_i(r) the ith orbital, \star is the complex conjugate.  These orbitals
 * are built out of atomic orbitals whose coefficients c_{k,l} minimize the HF or KS energy.
 *
 * The derivatives are taken via product rule and using symmetry and the fact that we are in the reals we have
 * a two factor at front:
 *
 * \frac{\partial \rho(r)}{\partial x} = 2 \sum_i w_i \sum_{k, l}  c_{k, i}^\star c_{j, i}
 *   \frac{\partial \phi_k(r)}{\partial x} \phi_l(r),
 *
 * The process to compute the gradient is as follows:
 * 1. Compute the derivative of the contractions shape (3, M, N)
 * 2. Compute the contractions (M, N)
 * 3. For each derivative, take the ith deriv of contractions and multiply it by the one-rdm.
 * 4. Do a hadamard product with the contractions.
 * 5. Take the sum.
 *
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] return_row If true, then return in row-major order (default), else return column-major.
 * @return Return the gradient of electron density of size (knumb_points, 3) in row-major order.
 */
__host__ std::vector<double> evaluate_electron_density_gradient(
    chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row = true
);

/**
 * Wrapper function over evaluate_electron_density_gradient, faster if cublas handler was initialized previously.
 */
__host__ std::vector<double> evaluate_electron_density_gradient_handle(
    // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row
);
}

#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_GRADIENT_CUH_
