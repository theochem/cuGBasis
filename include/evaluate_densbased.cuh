#ifndef GBASIS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_
#define GBASIS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_

#include <vector>

#include "iodata.h"

namespace gbasis {

/**
 * Compute the norm of the rows of a three-dimensional vector (e.g. the gradient) using GPU.
 *
 * @param h_points: Array in column-major order that stores the three-dimensional points.
 * @param knumb_pts:  Number of points in h_points.
 * @return A vector storing the norms of each row
 */
__host__ std::vector<double> compute_norm_of_3d_vector(double *h_points, int knumb_pts);


}
#endif //GBASIS_CUDA_INCLUDE_EVALUATE_DENSBASED_CUH_
