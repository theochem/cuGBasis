#ifndef GBASIS_CUDA_INCLUDE_PYMOLCULE_H_
#define GBASIS_CUDA_INCLUDE_PYMOLCULE_H_
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>

#include "cublas_v2.h"

#include "../include/iodata.h"

namespace py = pybind11;
using MatrixX3C = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
using MatrixX3R = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Vector3D = Eigen::Matrix<double, 1, 3>;
using IntVector3D =Eigen::Matrix<int, 1, 3>;
using Matrix33C =Eigen::Matrix<double, 3, 3, Eigen::ColMajor>;
using Matrix33R =Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

namespace gbasis {
/// Transfer basis-set object to constant memory.

class Molecule {
  std::string file_path;
  gbasis::IOData* iodata_;
  /// cublasHandle_t handle; Couldn't figure out how to fix this, only works once,

 public:
  /**
   * Construct Molecule class
   *
   * @param file_path: File path of the wavefunction file.
   */
  Molecule(const std::string &file_path);

  // Getters and Setters
  const std::string &getFilePath() const { return file_path; }
  const gbasis::IOData *getIOData() const { return iodata_; }

  // Methods
  void basis_set_to_constant_memory(bool do_segmented_basis);
  void clear_constant_memory();
  Vector compute_electron_density(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_electron_density_cubic(
      const Vector3D& klower_bnd, const Matrix33R& kaxes, const IntVector3D& knumb_points, const bool disp = false);
  Vector compute_laplacian(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_positive_definite_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_general_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points, const double alpha);
  MatrixX3R compute_electron_density_gradient(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_electrostatic_potential(const Eigen::Ref<MatrixX3R>&  points);
};

}
#endif //GBASIS_CUDA_INCLUDE_PYMOLCULE_H_
