#ifndef CHEMTOOLS_CUDA_INCLUDE_UTILS_H_
#define CHEMTOOLS_CUDA_INCLUDE_UTILS_H_

#include <array>
#include <cmath>
#include <utility>
#include <pybind11/numpy.h>

#include "./contracted_shell.h"

namespace chemtools {

typedef std::array<double, 3> CoordsXYZ;
typedef std::array<double, 9> ThreeDMatrixColOrder;
typedef std::array<int, 3> ShapeXYZ;

struct UniformGrid {
    CoordsXYZ l_bnd;
    CoordsXYZ u_bnd;
    ThreeDMatrixColOrder axes;

    UniformGrid(CoordsXYZ l, CoordsXYZ u, ThreeDMatrixColOrder ax) {
      l_bnd = l;
      u_bnd = u;
      axes = ax;
    }

    CoordsXYZ apply_axes_transformed(const CoordsXYZ& pt) {
      // Apply axes.T transformation to pt, i.e. pt.dot(axes.T)
      // Transforms the point back into the optimal cube
      double x = pt[0] * axes[0] + pt[1] * axes[3] + pt[2] * axes[6];
      double y = pt[0] * axes[1] + pt[1] * axes[4] + pt[2] * axes[7];
      double z = pt[0] * axes[2] + pt[1] * axes[5] + pt[2] * axes[8];
      return {x, y, z};
    }

    ShapeXYZ calculate_shape(CoordsXYZ spacing) {
      CoordsXYZ u_bnd_transf = apply_axes_transformed(u_bnd);
      CoordsXYZ l_bnd_transf = apply_axes_transformed(l_bnd);

      int x = static_cast<int>(std::ceil(std::fabs(u_bnd_transf[0] - l_bnd_transf[0]) / spacing[0]));
      int y = static_cast<int>(std::ceil(std::fabs(u_bnd_transf[1] - l_bnd_transf[1]) / spacing[1]));
      int z = static_cast<int>(std::ceil(std::fabs(u_bnd_transf[2] - l_bnd_transf[2]) / spacing[2]));
      return {x, y, z};
    }

    CoordsXYZ calculate_spacing(ShapeXYZ shape) {
      CoordsXYZ ut = apply_axes_transformed(u_bnd);
      CoordsXYZ lt = apply_axes_transformed(l_bnd);

      double sx = (ut[0] - lt[0]) / shape[0];
      double sy = (ut[1] - lt[1]) / shape[1];
      double sz = (ut[2] - lt[2]) / shape[2];
      return {sx, sy, sz};
    }

    ThreeDMatrixColOrder multiply_axes_by_spacing(double spacing_x, double spacing_y, double spacing_z) {
      // Multiplies the column with the spacing, i.e. first coloumn with spacing_x, second with spacing_y
      return {
          axes[0] * spacing_x, axes[1] * spacing_x, axes[2] * spacing_x,
          axes[3] * spacing_y, axes[4] * spacing_y, axes[5] * spacing_y,
          axes[6] * spacing_z, axes[7] * spacing_z, axes[8] * spacing_z
      };
    }
};

/**
 * Compute the lower and upper bound for the marching cubes of electron density.
 *
 * @param coordinates  The coordinates of each atom in the molecular system.
 * @param add_amount Optional, Amount to subtract/add to the lower/upper bound respectively, Default is zero.
 * @return The lower bound and upper bound of the molecule.
 */
std::pair<CoordsXYZ, CoordsXYZ> get_lower_and_upper_bounds(const double* coordinates, const int* charges,
                                                           int natoms, double add_amount=0);


/**
 * Compute the lower and axes for the grid defining the cube.
 *
 * @param[out] The origin of the grid.
 * @param[out] The axes (3x3 matrix) of the grid in column-major order.
 * @param coordinates  The coordinates of each atom in the molecular system in column major order
 * @param add_amount Optional, Amount to subtract/add to the lower/upper bound respectively, Default is zero.
 * @return The lower bound and upper bound of the molecule.
 */
UniformGrid get_grid_from_coordinates_charges( 
    const double* coordinates, const int* charges, int natoms, double add_amount
);

/// Convert an array to array_t without copying.
inline pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> as_pyarray_from_vector(std::vector<double> &seq) {
  // Took this from github/pybind/pybind11/issues/1042 from user YannickJadoul
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<std::vector<double>> seq_ptr = std::make_unique<std::vector<double>>(std::move(seq));
  auto capsule = pybind11::capsule(seq_ptr.get(), [](void *p) { std::unique_ptr<std::vector<double>>(reinterpret_cast<std::vector<double>*>(p)); });
  seq_ptr.release();
  return pybind11::array(size, data, capsule);
}
} // end chemtools
#endif //CHEMTOOLS_CUDA_INCLUDE_UTILS_H_
