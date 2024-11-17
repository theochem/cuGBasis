#ifndef CHEMTOOLS_CUDA_SRC_EVALUATE_PROMOLECULAR_CUH_
#define CHEMTOOLS_CUDA_SRC_EVALUATE_PROMOLECULAR_CUH_

#include <vector>
#include <string>
#include <unordered_map>

#define N 7500  // Number of blocks needs to be less than 64,000 kilobytes. 7500 = 60,000 kB/ 8 bytes for double

// Create a function pointer type definition for evaluating anything over contractions
//typedef void (*d_func_t)(double*, const double*, const int, const int, const int);


namespace chemtools {

// This points to the correct __device__ function that evaluates over contractions
//__device__ extern d_func_t p_evaluate_contractions;

/**
 * DEVICE FUNCTIONS
 * ---------------------------------------------------------------------------------------------------------------
 */
/**
 * Evaluate the promolecular density on any grid.
 *
 * Each thread is associated to the row/number of points in `d_density_array`.
 * Assumes that the atomic coordinates, numbers and promolecular model is stored in constant memory with the
 * global variable `g_constant_basis`.
 * See the file basis_to_gpu.cu for more information.
 *
 * @param[in, out] d_contractions_array  The device pointer to the contractions array of size (M, N) where M is
 *                  the number of contractions and N is the number of points. This is in row-major order.
 * @param d_points  The points in three-dimensions of shape (N, 3) stored in column-major order.
 * @param knumb_points  The number of points in the grid
 * @param knatom  Number of atoms to do that is in constant memory.
 */
__global__ void evaluate_promol_density_from_constant_memory_on_any_grid(
    double* d_density_array, const double* const d_points, const int knumb_points, const int index_atom_coords_start
);

/**
 * Evaluate the promolecular electrostatic on any grid
 * Same parameters as above
 */
__global__ void evaluate_promol_electrostatic_from_constant_memory_on_any_grid(
    double* d_density_array, const double* const d_points, const int knumb_points, const int index_atom_coords_start
);
/**
 * HOST FUNCTIONS
 * -----------------------------------------------------------------------
 */

/**
 * Evaluate the scalar property from promolecular model over any grid.
 *
 * @param[in] promol The promolecular information
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] property String containing what property it is, "density", "electrostatic", "laplacian"
 * @return h_density The promolecular density evaluated on each point.
 */
__host__ std::vector<double> evaluate_promol_scalar_property_on_any_grid(
    // promolecule information
    const double* const atom_coords,
    const long int *const atom_numbers,
    int natoms,
    const std::unordered_map<std::string, std::vector<double>>& promol_coeffs,
    const std::unordered_map<std::string, std::vector<double>>& promol_exps,
    const double* h_points,
    const int knumb_points,
    std::string property
    );


} // end chemtools
#endif //CHEMTOOLS_CUDA_SRC_EVALUATE_PROMOLECULAR_CUH_