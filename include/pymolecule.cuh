#ifndef CHEMTOOLS_CUDA_INCLUDE_PYMOLCULE_H_
#define CHEMTOOLS_CUDA_INCLUDE_PYMOLCULE_H_
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cublas_v2.h"

#include "../include/iodata.h"

namespace py = pybind11;
using MatrixX3C = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
using MatrixXXC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixX3R = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Vector3D = Eigen::Matrix<double, 1, 3>;
using IntVector = Eigen::Matrix<int, 1, Eigen::Dynamic>;
using IntVector3D = Eigen::Matrix<int, 1, 3>;
using Matrix33C = Eigen::Matrix<double, 3, 3, Eigen::ColMajor>;
using Matrix33R = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using TensorXXXR = Eigen::Tensor<double, 3, Eigen::RowMajor>;

namespace chemtools {
/// Transfer basis-set object to constant memory.

class Molecule {
  std::string file_path;
  chemtools::IOData* iodata_;
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
  const chemtools::IOData *getIOData() const { return iodata_; }
  const MatrixX3R getCoordinates() const;
  const IntVector getNumbers() const;

  // Methods
  void basis_set_to_constant_memory(bool do_segmented_basis);
  Vector compute_electron_density(const Eigen::Ref<MatrixX3R>&  points);
  MatrixXXC compute_molecular_orbitals(const Eigen::Ref<MatrixX3R>&  points);
  TensorXXXR compute_electron_density_cubic(
      const Vector3D& klower_bnd, const Matrix33R& kaxes, const IntVector3D& knumb_points, const bool disp = false);
  Vector compute_laplacian(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_positive_definite_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_general_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points, const double alpha);
  MatrixX3R compute_electron_density_gradient(const Eigen::Ref<MatrixX3R>&  points);
  TensorXXXR compute_electron_density_hessian(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_electrostatic_potential(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_norm_of_vector(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_reduced_density_gradient(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_weizsacker_ked(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_thomas_fermi_ked(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_general_gradient_expansion_ked(const Eigen::Ref<MatrixX3R>& array, double a, double b);
  Vector compute_empirical_gradient_expansion_ked(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_gradient_expansion_ked(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_general_ked(const Eigen::Ref<MatrixX3R>& array, const double a);
  Vector compute_hamiltonian_ked(const Eigen::Ref<MatrixX3R>& array);
  Vector compute_shannon_information_density(const Eigen::Ref<MatrixX3R>& array);
  };

}
#endif //CHEMTOOLS_CUDA_INCLUDE_PYMOLCULE_H_
