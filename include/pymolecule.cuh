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
#include <vector>

#include "cublas_v2.h"

#include "../include/iodata.h"

namespace py = pybind11;
using MatrixX3C = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
using MatrixXXC = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixX3R = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Vector3D = Eigen::Matrix<double, 1, 3>;
using IntVector = Eigen::Matrix<long int, 1, Eigen::Dynamic>;
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


class ProMolecule {
  MatrixX3R coord_atoms_;                                        // Coordinates of atoms (M, 3) in row-major order
  IntVector atnums_;                                             // Atomic number of atoms (M,)
  int natoms_;                                                   // Number of atoms M
  std::unordered_map<std::string, std::vector<double>> coeffs_;  // Keys: ELEMENT_PARAMETER_TYPE e.g. c_coeffs_s
  std::unordered_map<std::string, std::vector<double>> exps_;    // Mapping from element h_exps_p to exponents
  ProMolecule(
      const Eigen::Ref<MatrixX3R>& atom_coords,
      const Eigen::Ref<IntVector>& atom_numbers,
      int atom_length,
      const std::string file_path_to_data
      );

 public:
  // Had difficult creating py::init with pybind11 so decided to make my own create so that I don't have to
  //   do type declaration
  static ProMolecule create(
      const Eigen::Ref<MatrixX3R>& atom_coords,
      const Eigen::Ref<IntVector>& atom_numbers,
      int atom_length,
      const std::string file_path_to_data)
    {
      return ProMolecule(atom_coords, atom_numbers, atom_length, file_path_to_data);
    }

  // Functionality
  Vector compute_electron_density(const Eigen::Ref<MatrixX3R>&  points);
  MatrixX3R compute_electron_density_gradient(const Eigen::Ref<MatrixX3R>&  points);
  TensorXXXR compute_electron_density_hessian(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_laplacian(const Eigen::Ref<MatrixX3R>&  points);
  Vector compute_electrostatic_potential(const Eigen::Ref<MatrixX3R>&  points);

  // Getters
  MatrixX3R GetCoordAtoms() const {return coord_atoms_;}
  int GetNatoms() const {return natoms_;}
  const IntVector GetAtomicNumbers() const {return atnums_;}
  const std::unordered_map<std::string, std::vector<double>> GetPromolCoefficients() {return coeffs_;}
  const std::unordered_map<std::string, std::vector<double>> GetPromolExponents() {return exps_;}

  // Get the Coefficients Based on Element_Charge
  const std::vector<double>& GetCoefficients(const std::string& element_with_charge) {
    auto it = coeffs_.find(element_with_charge);
    if (it != coeffs_.end()) {
      return it->second;
    }
    else {
      throw std::out_of_range("Key not found in map .");
    }
  }
  // Get the Exponents Based on Element_Charge
  const std::vector<double>& GetExponents(const std::string& element_with_charge) {
    auto it = exps_.find(element_with_charge);
    if (it != exps_.end()) {
      return it->second;
    }
    else {
      throw std::out_of_range("Key not found in map. ");
    }
  }
};
}
#endif //CHEMTOOLS_CUDA_INCLUDE_PYMOLCULE_H_
