
#include <pybind11/pybind11.h>

#include "cublas_v2.h"

#include "../include/pymolecule.cuh"
#include "../include/basis_to_gpu.cuh"
#include "../include/evaluate_densbased.cuh"
#include "../include/evaluate_density.cuh"
#include "../include/evaluate_gradient.cuh"
#include "../include/evaluate_hessian.cuh"
#include "../include/evaluate_laplacian.cuh"
#include "../include/evaluate_kinetic_dens.cuh"
#include "../include/evaluate_electrostatic.cuh"

namespace py = pybind11;

chemtools::Molecule::Molecule(const std::string &file_path) {
  this->file_path = file_path;
  chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk(file_path);
  // Note that this copies the iodata object over the class, and so it envokes teh destructore
  this->iodata_ = new chemtools::IOData(iodata);
}

void chemtools::Molecule::basis_set_to_constant_memory(bool do_segmented_basis){
  // Convert from fchk file to IODATA object
  const chemtools::IOData* iodata = getIOData();

  // Transfer molecular basis to constant memory
  chemtools::MolecularBasis molecular_basis = iodata->GetOrbitalBasis();
  chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, do_segmented_basis, false);
}

const MatrixX3R chemtools::Molecule::getCoordinates() const {
  const double* pts_row_order = iodata_->GetCoordAtoms();
  MatrixX3R coordinates = Eigen::Map<const MatrixX3R>(pts_row_order, iodata_->GetNatoms(), 3);
  return coordinates;
}

const IntVector chemtools::Molecule::getNumbers() const {
  const int* atomic_numbers = iodata_->GetAtomicNumbers();
  IntVector atomic_numbs = Eigen::Map<const IntVector>(atomic_numbers, iodata_->GetNatoms());
  return atomic_numbs;
}

Vector chemtools::Molecule::compute_electron_density(const Eigen::Ref<MatrixX3R>&  points) {
  // Accept in row-major order because it is numpy default
  // Convert to column major order since it works better with the GPU code
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> dens = chemtools::evaluate_electron_density_on_any_grid(
      *iodata_, pts_col_order.data(), nrows
      );
  Vector v2 = Eigen::Map<Vector>(dens.data(), nrows);
  return v2;
}


TensorXXXR chemtools::Molecule::compute_electron_density_cubic(
    const Vector3D& klower_bnd, const Matrix33R& kaxes, const IntVector3D& knumb_points, const bool disp
) {
  Matrix33C kaxes_col_order = kaxes;
  size_t numb_pts = knumb_points[0] * knumb_points[1] * knumb_points[2];
  std::vector<double> dens = chemtools::evaluate_electron_density_on_cubic(
      *iodata_, {klower_bnd[0], klower_bnd[1], klower_bnd[2]}, kaxes_col_order.data(),
      {knumb_points[0], knumb_points[1], knumb_points[2]}, disp
  );
  /// Eigen Tensor doesn't work with pybind11, so the trick here would be to use array_t to convert them
  TensorXXXR v2 = Eigen::TensorMap<TensorXXXR>(dens.data(), knumb_points[0], knumb_points[1], knumb_points[2]);
  return v2;
}


MatrixX3R chemtools::Molecule::compute_electron_density_gradient(const Eigen::Ref<MatrixX3R>&  points) {
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> grad = chemtools::evaluate_electron_density_gradient(
      *iodata_, pts_col_order.data(), nrows
  );
  MatrixX3R v2 = Eigen::Map<MatrixX3R>(grad.data(), nrows, 3);
  return v2;
}

TensorXXXR chemtools::Molecule::compute_electron_density_hessian(const Eigen::Ref<MatrixX3R>&  points) {
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> hessian_row = chemtools::evaluate_electron_density_hessian(
      *iodata_, pts_col_order.data(), nrows, true
  );
  /// Eigen Tensor doesn't work with pybind11, so the trick here would be to use array_t to convert them
  TensorXXXR v2 = Eigen::TensorMap<TensorXXXR>(hessian_row.data(), nrows, 3, 3);
  return v2;
}

Vector chemtools::Molecule::compute_laplacian(const Eigen::Ref<MatrixX3R>&  points) {
  // Accept in row-major order because it is numpy default
  // Convert to column major order since it works better with the GPU code
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> laplacian = chemtools::evaluate_laplacian(
      *iodata_, pts_col_order.data(), nrows
  );
  Vector v2 = Eigen::Map<Vector>(laplacian.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_positive_definite_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points) {
  // Accept in row-major order because it is numpy default
  // Convert to column major order since it works better with the GPU code
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> kinetic_dens = chemtools::evaluate_positive_definite_kinetic_density(
      *iodata_, pts_col_order.data(), nrows
  );
  Vector v2 = Eigen::Map<Vector>(kinetic_dens.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_general_kinetic_energy(const Eigen::Ref<MatrixX3R>&  points, const double alpha) {
  MatrixX3C pts_col_order = points;
  size_t nrows = points.rows();
  std::vector<double> kinetic_dens = chemtools::evaluate_general_kinetic_energy_density(
      *iodata_, alpha, pts_col_order.data(), nrows
  );
  Vector v2 = Eigen::Map<Vector>(kinetic_dens.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_electrostatic_potential(const Eigen::Ref<MatrixX3R>&  points) {
  // Accept in row-major order because it is numpy default
  // Convert to column major order since it works better with the GPU code
  MatrixX3R pts_row_order = points;
  size_t nrows = points.rows();
  std::vector<double> esp = chemtools::compute_electrostatic_potential_over_points(
      *iodata_, pts_row_order.data(), nrows, 1e-11, false
      );
  Vector v2 = Eigen::Map<Vector>(esp.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_norm_of_vector(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> norm = chemtools::compute_norm_of_3d_vector(pts_row_order.data(), nrows);
  Vector v2 = Eigen::Map<Vector>(norm.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_reduced_density_gradient(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> reduced = chemtools::compute_reduced_density_gradient(*iodata_, pts_row_order.data(), nrows);
  Vector v2 = Eigen::Map<Vector>(reduced.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_weizsacker_ked(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> w_ked = chemtools::compute_weizsacker_ked(*iodata_, pts_row_order.data(), nrows);
  Vector v2 = Eigen::Map<Vector>(w_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_thomas_fermi_ked(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> tf_ked = chemtools::compute_thomas_fermi_energy_density(*iodata_, pts_row_order.data(), nrows);
  Vector v2 = Eigen::Map<Vector>(tf_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_general_gradient_expansion_ked(
    const Eigen::Ref<MatrixX3R>& array, const double a, const double b
    ){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> tf_ked = chemtools::compute_ked_gradient_expansion_general(
      *iodata_, pts_row_order.data(), nrows, a, b
      );
  Vector v2 = Eigen::Map<Vector>(tf_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_empirical_gradient_expansion_ked(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> e_ked = chemtools::compute_ked_gradient_expansion_general(
      *iodata_, pts_row_order.data(), nrows, 1.0 / 5.0, 1.0 / 6.0
  );
  Vector v2 = Eigen::Map<Vector>(e_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_gradient_expansion_ked(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> e_ked = chemtools::compute_ked_gradient_expansion_general(
      *iodata_, pts_row_order.data(), nrows, 1.0 / 9.0, 1.0 / 6.0
  );
  Vector v2 = Eigen::Map<Vector>(e_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_general_ked(const Eigen::Ref<MatrixX3R>& array, const double a){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> gen_ked = chemtools::compute_general_ked(
      *iodata_, pts_row_order.data(), nrows, a
  );
  Vector v2 = Eigen::Map<Vector>(gen_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_hamiltonian_ked(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> ham_ked = chemtools::compute_general_ked(
      *iodata_, pts_row_order.data(), nrows, 0.0
  );
  Vector v2 = Eigen::Map<Vector>(ham_ked.data(), nrows);
  return v2;
}

Vector chemtools::Molecule::compute_shannon_information_density(const Eigen::Ref<MatrixX3R>& array){
  MatrixX3R pts_row_order = array;
  size_t nrows = array.rows();
  std::vector<double> entropy = chemtools::compute_shannon_information_density(
      *iodata_, pts_row_order.data(), nrows
  );
  Vector v2 = Eigen::Map<Vector>(entropy.data(), nrows);
  return v2;
}