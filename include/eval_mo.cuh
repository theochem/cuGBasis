
#ifndef CUGBASIS_SRC_EVAL_MO_CUH_
#define CUGBASIS_SRC_EVAL_MO_CUH_

#include "contracted_shell.h"
#include "iodata.h"

namespace chemtools {

/**
 * Evaluate the molecular orbitals from storing molecular basis in constant memory over any grid.
 *
 * @param[in] iodata IOData object
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] spin String inThe type of occupied spin orbitals. Options are "a" (for alpha), "b" (for beta), and
           "ab" (for alpha + beta).
 * @return h_mol_orbitals The molecular orbitals (M, N) coloumn-order evaluated on each point.
 */
__host__ std::vector<double> eval_MOs(
    IOData& iodata, const double* h_points, int n_pts, const std::string& spin = "ab"
);


/**
 * Evaluate the molecular orbitals derivatives.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h__mol_orbitals_deriv The first derivative molecular orbitals order (3, N, M).
 */
__host__ std::vector<double> eval_MOs_derivs(
    IOData& iodata, const double* h_points, const int n_pts
);


/**
 * Evaluate the molecular orbitals second derivatives.
 *
 * @param[in] h_points Array in column-major order that stores the `N` three-dimensional points.
 * @param[in] n_pts Number of points in d_points.
 * @param[in] nbasisfuncs The number of basis functions.
 * @return h_sec_mol_orbitals The second derivatives MO order (6, N, M)
 */
__host__ std::vector<double> eval_MOs_second_derivs(
    IOData& iodata, const double* h_points, const int n_pts
);
}
#endif //CUGBASIS_SRC_EVAL_MO_CUH_
