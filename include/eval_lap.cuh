#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_LAPLACIAN_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_LAPLACIAN_CUH_

#include "cublas_v2.h"

#include "iodata.h"


namespace chemtools {

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
 
/**
 * Device helper function for evaluating the sum of second derivative of each contractions
 *
 * Assumes basis set information is stored in GPU constant memory via basis_to_gpu.cu file.
 *
 * @param[in, out] d_lap The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 *                  Values should be set to zero before using.
 * @param pt The 4f grid point
 * @param n_pts  The number of points.
 * @param idx  The global index of the thread, which also corresponds to the point.
 * @param iorb_start Where to start computing the atomic orbitals, see eval.cu
 */
__device__ __forceinline__ void eval_AOs_lap(
    double *d_lap,
    const double3& pt,
    const int &n_pts,
    uint &idx,
    const int &iorb_start);


__global__ void eval_AOs_lap_from_constant_memory_on_any_grid(
          double* __restrict__ d_lap,
    const double* __restrict__ d_points,
    const int     n_pts,
    const int     n_cshells,
    const int     iorb_start
);


/**
 * HOST FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Evaluate sum of second derivatives of the contractions.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return sum of second derivatives of each contraction this is in row-order (M, N),
 *          where M =number of basis-functions.
 */
__host__ std::vector<double> evaluate_sum_of_second_derivative_contractions(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Helper function for computing the Laplacian.
 *
 *   2 \sum_i \sum_j c_{i, j}  [\sum_k \partial \phi_i^2 \ \partial x_k^2] \phi_j .
 *
 *   where \phi_i is the ith contraction and c_{i, j} are the MO coefficients.
 *
 * @param handle: Cublas handle
 * @param iodata:  The IOData object that stores the molecules basis.
 * @param h_laplacian: Vector whose values are to be replaced.
 * @param h_points: Array in column-major order that stores the three-dimensional points.
 * @param knumb_points: Number of points
 * @param knbasisfuncs: Number of basis-functions.
 */
__host__ void compute_first_term(
    const cublasHandle_t& handle, const chemtools::MolecularBasis& basis, std::vector<double>& h_laplacian,
    const double* const h_points, const int knumb_points, const int knbasisfuncs, const double* const h_one_rdm
);

__host__ std::vector<double> evaluate_laplacian_on_any_grid_handle(
    cublasHandle_t& handle, IOData &iodata, const double* h_points, const int n_pts, const std::string& spin = "ab"
);

/**
 * Evaluate the Laplacian of the electron density on a grid of points.
 *
 * The Laplacian is the sum of the second derivatives of the electron density.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] type Type of occupied spin orbitals which can be either "a" (for alpha), "b" (for
                        beta), and "ab" (for alpha + beta).
 * @return h_laplacian The Laplacian of the electron density evaluated on each point.
 */
__host__ std::vector<double> evaluate_laplacian(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points, const std::string& type = "ab"
);
}
#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_LAPLACIAN_CUH_
