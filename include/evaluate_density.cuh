#ifndef CHEMTOOLS_CUDA_SRC_EVALUATE_DENSITY_CUH_
#define CHEMTOOLS_CUDA_SRC_EVALUATE_DENSITY_CUH_

#include "cublas_v2.h"

#include "contracted_shell.h"
#include "iodata.h"

#define N 7500  // Number of blocks needs to be less than 64,000 kilobytes. 7500 = 60,000 kB/ 8 bytes for double

// Create a function pointer type definition for evaluating anything over contractions
typedef void (*d_func_t)(double*, const double*, const int, const int, const int);


namespace chemtools {

// This points to the correct __device__ function that evaluates over contractions
__device__ extern d_func_t p_evaluate_contractions;

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Device helper function for evaluating a contractions on a single point (corresponding to a single thread).
 *
 * @param[in, out] d_contractions_array The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 * @param grid_x The grid point in the x-axis.
 * @param grid_y  The grid point in the y-axis.
 * @param grid_z  The grid point in the z-axis.
 * @param knumb_points  The number of points.
 * @param global_index  The global index of the thread, which also corresponds to the point.
 * @param i_contr_start Index on where it starts to update the rows/contractions in `d_contractions_array`. Should
 *                      match what's in constant memory.
 */
__device__ void evaluate_contractions(
    double* d_contractions_array, const double& grid_x, const double& grid_y, const double& grid_z,
    const int& knumb_points, unsigned int& global_index, const int& i_contr_start = 0
);

/**
 * Evaluate contractions from constant memory.
 *
 * Each thread is associated to the row/number of points in `d_contractions_array`. This is in row-major order.
 *
 * @param[in,out] g_d_constant_basis Global pointer to the baiss set information stored in constant memory.
 * @param[out] d_contractions_array  The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 *                  Values should be set to zero before using.
 * @param[in] d_klower_bnd The lower bound (bottom, left-most, down-most point) of the grid.
 * @param[in] d_axes_array_col1 The first column of axes of the grid defining the directions of the coordinate system
 * @param[in] d_axes_array_col2 The second column of axes of the grid defining the direction of the coordinate system
 * @param[in] d_axes_array_col3 The third column of axes of the grid defining the direction of the coordinate system
 * @param[in] d_knumb_points The number of points in each direction of the cubic grid.
 */
__global__ void evaluate_contractions_from_constant_memory_on_cubic_grid(
    double* d_contractions_array, const double3 d_klower_bnd, const double3 d_axes_array_col1,
    const double3 d_axes_array_col2, const double3 d_axes_array_col3, const int3 d_knumb_points
);

/**
 * Evaluate contractions on any grid.
 *
 * Each thread is associated to the row/number of points in `d_contractions_array`. This is in row-major order.
 * Assumes that the basis set information is stored in constant memory with the global variable `g_constant_basis`.
 * See the file basis_to_gpu.cu for more information.
 *
 * @param[in, out] d_contractions_array  The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 * @param d_points  The points in three-dimensions of shape (N, 3) stored in column-major order.
 * @param knumb_points  The number of points in the grid
 * @param knumb_contractions  Not needed but total number of contractions to update d_contractions_array.
 * @param i_const_start   Not needed but index of where to start updating the contractions over the rows M.
 */
__device__ void evaluate_contractions_from_constant_memory_on_any_grid(
    double* d_contractions_array, const double* const d_points, const int knumb_points, const int knumb_contractions,
    const int i_const_start = 0
);


/**
 * HOST FUNCTIONS
 * -----------------------------------------------------------------------
 */

/**
 * Evaluate the electron density from storing molecular basis in constant memory in cubic grid format.
 *
 * @param[out] d_electron_density Pointer to the electron density in the device memory.
 * @param[in] kaxes_col The  axes of the grid defining the directions of the coordinate system in column-major order.
 * @param[in] disp Display the output in certain calculations.
 */
__host__ std::vector<double> evaluate_electron_density_on_cubic(
    chemtools::IOData& iodata, const double3 klower_bnd, const double* kaxes_col, const int3 knumb_points, const bool disp = false
);

/**
 * Evaluate the electron density from storing molecular basis in constant memory over any grid.
 *
 *
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_electron_density The electron density evaluated on each point.
 */
__host__ std::vector<double> evaluate_electron_density_on_any_grid(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
    );

/**
 * Evaluate the electron density from storing molecular basis in constant memory over any grid.
 *
 *
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_electron_density The electron density evaluated on each point.
 */
__host__ std::vector<double> evaluate_electron_density_on_any_grid_handle(
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Evaluate the molecular orbitals from storing molecular basis in constant memory over any grid.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_mol_orbitals The molecular orbitals (M, N) coloumn-order evaluated on each point.
 */
__host__ std::vector<double> evaluate_molecular_orbitals_on_any_grid(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Evaluate the molecular orbitals from storing molecular basis in constant memory over any grid.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_mol_orbitals The molecular orbitals (M, N) coloumn-order evaluated on each point.
 */
__host__ std::vector<double> evaluate_molecular_orbitals_on_any_grid_handle(
    cublasHandle_t& handle, chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

} // end chemtools
#endif //CHEMTOOLS_CUDA_SRC_EVALUATE_CONTRACTIONS_CUH_