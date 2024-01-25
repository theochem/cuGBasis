#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_ELECTROSTATIC_H_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_ELECTROSTATIC_H_

#include "./iodata.h"

namespace chemtools {
/**
 * Calculates point charge integrals integrals from basis set stored in constant memory.
 *
 * @param[out] d_point_charge Contains the point charge integral, has size (M(M+1)/2, K), where M is the number of
 *                            basis functions and K is the number of grid points. The intgrals M(M+1)/2 of each
 *                            grid point is stored in (upper stored) triangular packed format and this array is stored in 
 *                            row-major.
 * @param[in] d_grid Linear array holding the three-dimensional points, in row-major order.
 * @param[in] knumb_pts The number of points K.
 * @param[in] nbasis The number of basis functions M.
 * @param[in] screen_tol Tolerance for integral screening for primitive integrals.
 * @note Each thread corresponds to a grid point and does all of the integrals. Most of the code is python-generated
 *       and uses template meta-programming for the integral recursions. This assumes that the basis set is
 *       decontracted (i.e. each contracted shell only has one shell) and is in constant memory.
 */
__global__ void compute_point_charge_integrals(
    double *d_point_charge, const double *const d_grid, const int knumb_pts, const int nbasis, const double screen_tol);

/**
 * Calculates electrostatic potential from basis set stored in constant memory.
 *
 * @param[in] iodata IOdata object that stores the molecular and basis set information.
 * @param[in] grid Array of three-dimensional grid points stored in row-major order.
 * @param[in] knumb_pts The number of grid points.
 * @param[in] screen_tol Tolerance for integral screening for primitive integrals.
 * @param[in] disp Display the printing of each step.
 * @returns Array of electrostatic potential evaluated at the grid points. Length is the number of grid points.
 *
 * @notes The point charge integrals uses the GPU. The one rdm hadamard product also uses the GPU.
 */
__host__ std::vector<double> compute_electrostatic_potential_over_points(
    chemtools::IOData &iodata, double *gridm, int knumb_pts, const double screen_tol=1e-11, const bool disp = false);

}
#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_ELECTROSTATIC_H_
