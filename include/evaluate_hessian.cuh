#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_HESSIAN_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_HESSIAN_CUH_

#include "cublas_v2.h"

#include "contracted_shell.h"
#include "iodata.h"

// Create a function pointer type definition
typedef void (*d_func_t)(double*, const double*, const int, const int, const int);


namespace chemtools {

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
// This points to the correct __device__ function that evaluates over contractions
__device__ extern d_func_t p_evaluate_sec_contractions;

/**
 * Evaluate the second derivative of contractions from any grid of points.
 *
 * Assumes basis set information is stored in GPU constant memory via basis_to_gpu.cu file.
 *
 * The calculation of the derivatives is given by the file "./generate_hessian_cont.py".
 * The derivatives are stored with the order "xx, xy, xz, yy, yz, zz".
 *
 * @param[in,out] g_d_constant_basis Pointer to the the output.
 * @param[out] d_sec_deriv_contracs  The device pointer to the second contractions array of size (6, M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order, i.e.
 *                  Derivatives first, then contractions then points.
 *                  Values should be set to zero before using.
 * @param[in] d_points The device pointer to the grid points of size (N, 3) stored in column-major order.
 * @param[in] d_knumb_points The number of points in the grid.
 */
__device__ void evaluate_sec_deriv_contractions_from_constant_memory(
    double* d_sec_deriv_contracs, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_contr_start = 0
);


/**
 * HOST FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Return second derivatives of the contractions to the host
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Assumes this is in column-major order
 * @param[in] knumb_points  Number of points in d_points.
 * @return Return the second derivatives of contractions in order (6, M, N) row-major order.
 */
__host__ std::vector<double> evaluate_contraction_second_derivative(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Helper function for computing the first term:
 *
 * \sum_{k, l}  c_{k, l} \bigg[
 *          \frac{\partial \phi_k(r)}{\partial x_n} \frac{\partial \phi_l(r)}{\partial x_m}
 */
__host__ static void evaluate_first_term(
    cublasHandle_t& handle, const chemtools::MolecularBasis& basis, double* d_hessian, const double* const d_points,
    const double* const h_one_rdm, const size_t& numb_pts_iter, const size_t& knbasisfuncs
);

/**
 * Evaluate the Hessian of the electron density from molecular basis in constant memory.
 *
 * The derivatives are taken via product rule and using symmetry and the fact that we are in the reals we have
 * a two factor at front:
 *
 * \frac{\partial^2 \rho(r)}{\partial x_n \partial x_m} = 2 \sum_{k, l}  c_{k, l} \bigg[
 *          \frac{\partial \phi_k(r)}{\partial x_n} \frac{\partial \phi_l(r)}{\partial x_m} + \
 *          \phi_l(r) \frac{\partial^2 \phi_k(r)}{\partial x_n \partial x_m}
 *          \bigg],
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] return_row If true, then return in row-major order (default), else return column-major.
 * @return Return the Hessian of electron density of size (knumb_points, 9) in row-major order.
 */
__host__ std::vector<double> evaluate_electron_density_hessian(
    chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row = true
);

/**
 * Wrapper function, faster if cublas handler was initialized previously.
 */
__host__ std::vector<double> evaluate_electron_density_hessian_handle(
    // Initializer of cublas and set to prefer L1 cache ove rshared memory since it doens't use it.
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, int knumb_points, bool return_row
);
}

#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_HESSIAN_CUH_
