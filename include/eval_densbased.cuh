#ifndef CHEMTOOLS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_

#include <vector>

#include "iodata.h"

namespace chemtools {

/**
 * Compute the norm of the rows of a three-dimensional vector (e.g. the gradient) using GPU.
 *
 * @param h_points: Array in column-major order that stores the three-dimensional points.
 * @param knumb_pts:  Number of points in h_points.
 * @return A vector storing the norms of each row
 */
__host__ std::vector<double> compute_norm_of_3d_vector(double *h_points, int knumb_pts);

/**
 * Compute the reduced density gradient
 *
 * Note: that it uses GPU to compute the density and gradient on the GPU, transfers
 *  it to host, then transfers it back to the GPU to divide. It may be most optimal
 *  for the user to just compute it themselves using Python.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return the reduced density gradient.
 */
__host__ std::vector<double> compute_reduced_density_gradient(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
    );

/**
 * Compute the Von Weizsacker Kinetic Energy Density
 *
 * Note: that it uses GPU to compute the density and gradient on the GPU, transfers
 *  it to host, then transfers it back to the GPU to divide. It may be most optimal
 *  for the user to just compute it themselves using Python.
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return the reduced density gradient.
 */
__host__ std::vector<double> compute_weizsacker_ked(
    chemtools::IOData& iodata, const double* h_points, const int knumb_points
);

/**
 * Compute the Thomas-Fermi kinetic energy density.
 *
 * \frac{3}{10} (3\pi)^{2/3} \rho(\mathbf{r})^{5/3}
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return the Thomas-Fermi kinetic energy density
 */
__host__ std::vector<double> compute_thomas_fermi_energy_density(
    chemtools::IOData& iodata, const double* h_points, int knumb_points
);

/**
 * Compute the general gradient expansion approximation of kinetic energy density
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] a, b: Parameters of the general gradient expansion approximation
 * @return Return the general gradient expansion approximation of kinetic energy density
 */
__host__ std::vector<double> compute_ked_gradient_expansion_general(
    chemtools::IOData& iodata, const double* h_points, int knumb_points, double a, double b
);

/**
 * Compute general(ish) kinetic energy density.
 *
 * \tau_G(\mathbf{r}, \alpha)  = \tau_{PD}(\mathbf{r}) + \frac{1}{4} (\alpha - 1) \nabla^2 \rho(\mathbf{r}),
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @param[in] a: Parameters
 * @return Return general(ish) kinetic energy density.
 */
__host__ std::vector<double> compute_general_ked(
    chemtools::IOData& iodata, const double* h_points, int knumb_points, double a
);

/**
 * Compute Shannon information density
 *
 * s(\mathbf{r}) = -\rho(\mathbf{r}) \log \rho(\mathbf{r}),
 *
 * @param[in] iodata  The IOData object that stores the molecules basis.
 * @param[in] h_points Array in column-major order that stores the three-dimensional points.
 * @param[in] knumb_points Number of points in d_points.
 * @return Return Shannon information density
 */
__host__ std::vector<double> compute_shannon_information_density(
    chemtools::IOData& iodata, const double* h_points, int knumb_points
);
}
#endif //CHEMTOOLS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_
