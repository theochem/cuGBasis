
#ifndef CUGBASIS_SRC_EVAL_AO_CUH_
#define CUGBASIS_SRC_EVAL_AO_CUH_

#include "contracted_shell.h"
#include "iodata.h"

namespace chemtools {
/**
 * Evaluate the atomic orbitals from storing atomic basis in constant memory over any grid.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_AO The atomic orbitals (M, N) coloumn-order evaluated on each point.
 */
__host__ std::vector<double> eval_AOs(
    IOData& iodata, const double* h_points, const int n_pts
);


/**
 * Evaluate the atomic orbitals derivatives.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h__AO_deriv The first derivative atomic orbitals order (3, N, M).
 */
__host__ std::vector<double> eval_AOs_derivs(
    IOData& iodata, const double* h_points, const int n_pts
);


/**
 * Evaluate the atomic orbitals second derivatives.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_AO_sec The second derivatives MO order (6, N, M)
 */
__host__ std::vector<double> eval_AOs_second_derivs(
    IOData& iodata, const double* h_points, const int n_pts
);

}
#endif //CUGBASIS_SRC_EVAL_AO_CUH_
