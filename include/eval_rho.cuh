#ifndef CHEMTOOLS_CUDA_SRC_EVALUATE_DENSITY_CUH_
#define CHEMTOOLS_CUDA_SRC_EVALUATE_DENSITY_CUH_

#include "cublas_v2.h"

#include "contracted_shell.h"
#include "iodata.h"


namespace chemtools {

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Device helper function for evaluating a contractions on a single point (corresponding to a single thread).
 *
 * @param[in, out] d_AO_vals The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 * @param grid_x The grid point in the x-axis.
 * @param grid_y  The grid point in the y-axis.
 * @param grid_z  The grid point in the z-axis.
 * @param n_pts  The number of points.
 * @param idx  The global index of the thread, which also corresponds to the point.
 * @param i_contr_start Index on where it starts to update the rows/contractions in `d_AO_vals`. Should
 *                      match what's in constant memory.
 */
__device__ __forceinline__ void eval_AOs(
    double* d_AO_vals, const double3& pt, const int& n_pts, unsigned int& idx, const int& iorb_start = 0
);


/**
 * Evaluate contractions on any grid.
 *
 * Each thread is associated to the row/number of points in `d_AO_vals`. This is in row-major order.
 * Assumes that the basis set information is stored in constant memory with the global variable `g_constant_basis`.
 * See the file basis_to_gpu.cu for more information.
 *
 * @param[in, out] d_AO_vals  The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 * @param d_points  The points in three-dimensions of shape (N, 3) stored in column-major order.
 * @param n_pts  The number of points in the grid
 * @param n_cshells  Not needed but total number of contractions to update d_AO_vals.
 * @param iorb_start   Not needed but index of where to start updating the contractions over the rows M.
 */
__global__ __launch_bounds__(128) void eval_AOs_from_constant_memory_on_any_grid(
    double* __restrict__ d_AO_vals, const double* __restrict__ d_points, const int n_pts, const int n_cshells,
    const int iorb_start = 0
);


/**
 * HOST FUNCTIONS
 * -----------------------------------------------------------------------
 */

/**
 * Evaluate the electron density from storing molecular basis in constant memory over any grid.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] type Type of occupied spin orbitals which can be either "a" (for alpha), "b" (for
           beta), and "ab" (for alpha + beta).
 * @return h_electron_density The electron density evaluated on each point.
 */
__host__ std::vector<double> evaluate_electron_density_on_any_grid(
    chemtools::IOData& iodata, const double* h_points, const int n_pts, const std::string& spin = "ab"
    );

/**
 * Evaluate the electron density from storing molecular basis in constant memory over any grid.
 *
 *
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @param[in] spin String inThe type of occupied spin orbitals. Options are "a" (for alpha), "b" (for beta), and
               "ab" (for alpha + beta).
 * @return h_electron_density The electron density evaluated on each point.
 */
__host__ std::vector<double> evaluate_electron_density_on_any_grid_handle(
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int n_pts, const std::string& spin
);

} // end chemtools
#endif //CHEMTOOLS_CUDA_SRC_EVALUATE_CONTRACTIONS_CUH_