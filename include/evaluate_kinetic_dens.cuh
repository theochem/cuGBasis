

#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_KINETIC_DENS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_KINETIC_DENS_CUH_

#include "cublas_v2.h"

#include "iodata.h"

namespace chemtools {
__host__ std::vector<double> evaluate_pos_def_kinetic_density_on_any_grid_handle(
    cublasHandle_t &handle, chemtools::IOData &iodata, const double *h_points, const int knumb_points
);

/**
 * Evaluate the positive definite kinetic energy density on a grid of points.
 *
 *      t_+(r) = [\frac{1}{2}] \nabla_r \dot \nabla_{r^\prime} \gamma(r, r^\prime),
 *
 *  where \gamma is the one-electron reduced density matrix.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return h_kinetic_energy_density The kinetic energy density evaluated on each point.
 */
__host__ std::vector<double> evaluate_positive_definite_kinetic_density(
    chemtools::IOData &iodata, const double *h_points, const int knumb_points
);


/**
 * Evaluate the general kinetic energy density on a grid of points.
 *
 *      t_\alpha(r) = t_+(r) + \alpha \nabla^2 \rho(r)
 *
 *  where t_+(r) is the positive definite kinetic energy density and \nabla^2 is the Laplacian.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] alpha Parameter of the general form of the kinetic energy density.
 * @param[in] knumb_points Number of points in d_points.
 * @return h_gen_kinetic_density The general form of the kinetic energy density evaluated on each point.
 */
__host__ std::vector<double> evaluate_general_kinetic_energy_density(
    chemtools::IOData &iodata, const double alpha, const double *h_points, const int knumb_points
);
} // chemtools
#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_KINETIC_DENS_CUH_
