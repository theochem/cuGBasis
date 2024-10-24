#include <exception>
#include <list>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include "../include/iodata.h"

/**
 *  Shell types (NShell values): 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
 *
 */
chemtools::IOData chemtools::get_molecular_basis_from_fchk(const std::string& fchk_file_path, bool disp) {
  // Start python interpreter.
  py::module_ iodata = py::module_::import("iodata");

  // Load the iodata object.
  py::object iodata_obj = iodata.attr("load_one")(fchk_file_path);

  // Load the orbital basis information.
  py::object orbital_basis = iodata_obj.attr("obasis");

  // Make sure primitive normalization is set to "L2".
  std::string primitive_normalization = orbital_basis.attr("primitive_normalization").cast<std::string>();
  if(primitive_normalization != "L2"){
    throw std::runtime_error("Primitive normalization needs to be L2.");
  }

  // Get number of basis functions.
  const int nshells = orbital_basis.attr("nbasis").cast<int>();
  if (disp) {
    printf("Number of basis functions is %d \n", nshells);
  }

  // Obtain the Generalized Contraction Shell Information.
  py::object shells = orbital_basis.attr("shells");
  int n_cont_shells = shells.attr("__len__")().cast<int>();
  if (disp) {
    printf("Number of contracted shells is %d \n", n_cont_shells);
//  py::print(orbital_basis.attr("conventions"));
  }

  // Get Atomic Coordinates Information.
  std::vector<std::array<double, 3>> all_coordinates;
  py::list atcoords = iodata_obj.attr("atcoords");
  int numb_coordinates = atcoords.size();
  py::array_t<double, py::array::c_style | py::array::forcecast> coords;
  for(int i = 0; i < numb_coordinates; i++){
    coords = atcoords.attr("__getitem__")(i);
    all_coordinates.push_back({coords.at(0), coords.at(1), coords.at(2)});
  }

  // Get the Shell Information.
  std::vector<chemtools::GeneralizedContractionShell> molecular_basis;
  for(int i = 0; i < n_cont_shells; i++) {
    py::object contracted_shell = shells.attr("__getitem__")(i);
    if (disp) {
      py::print(contracted_shell);
    }
    // Get Coordinate
    int icenter = contracted_shell.attr("icenter").cast<int>();
    std::array<double, 3> coordinate = all_coordinates[icenter];

    // Get Angular momentum.
    py::list py_angmoms = contracted_shell.attr("angmoms");
    py::list py_kinds = contracted_shell.attr("kinds");
    std::vector<int> angmoms;
    for(int j = 0; j < py_angmoms.size(); j++){
      int angmom = py_angmoms[j].cast<int>();
      char kind = py_kinds[j].cast<char>();
      if(kind == 'c') {
        angmoms.push_back(angmom);
      }
      else if (kind == 'p') {
        if (angmom == 0) {
          angmoms.push_back(0);
        }
        else if (angmom == 1) {
          angmoms.push_back(1);
        }
        else if (angmom >= 2){
          // If it is a D-orbital, then return -2 to follow how .fchk Gaussian work.
          // Recall that D pure type orbitals is different from Cartesian type orbitals.
          angmoms.push_back(-angmom);
        }
        else {
          throw std::runtime_error("There is no recognizable angular momentum term.");
        }
      }
      else {
        throw std::runtime_error("No recognition of the kind " + kind);
      }
    }
    if (disp) {
      printf("Angular momentums \n ");
      for(int a : angmoms){
        printf("Ang mom %d  ", a);
      }
      printf("\n");
    }

    // Get Exponents
    std::vector<double> exponents;
    py::array_t<double, py::array::c_style | py::array::forcecast> py_exponent = contracted_shell.attr("exponents").cast<
        py::array_t<double, py::array::c_style | py::array::forcecast>>();
    for(int j = 0; j < py_exponent.size(); j++) {
      exponents.push_back(py_exponent.at(j));
    }


    // Get Coefficients The array containing the coefficients of the normalized primitives in each contraction;
    //        shape = (nprim, ncon).
    std::vector<std::vector<double>> coefficients;
    py::array_t<double, py::array::c_style | py::array::forcecast> py_coeffs = contracted_shell.attr("coeffs").
        cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    int ncom = angmoms.size();
    int nprim = exponents.size();
    for(int j = 0; j < ncom; j++) {
      std::vector<double> coefficients_j;
      for(int k = 0; k < nprim; k++) {
        coefficients_j.push_back(py_coeffs[py::make_tuple(k, j)].cast<double>());
      }
      coefficients.push_back(coefficients_j);
    }

    molecular_basis.emplace_back(GeneralizedContractionShell {angmoms, coordinate, exponents, coefficients});

    if(disp){
      printf("\n");
    }
  }



  // From IOData: Dictionary where keys are names and values are one-particle density matrices.
  // Names can be scf, post_scf, scf_spin, post_scf_spin. These matrices are always expressed in the AO basis.
  py::dict one_rdms = iodata_obj.attr("one_rdms");
  py::array_t<double, py::array::c_style> one_rdm;
  if (! one_rdms.empty()) {
    one_rdm = one_rdms["scf"].cast<py::array_t<double>>();
  }

  // Get Molecular orbital coefficients and occupations.
  py::object molecular_orb_obj = iodata_obj.attr("mo");
  py::array_t<double, py::array::c_style> coeffs = (molecular_orb_obj.attr("coeffs")).cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> occs = molecular_orb_obj.attr("occs").cast<py::array_t<double>>();
  int one_rdm_shape_row = coeffs.shape()[0];
  int one_rdm_shape_col = coeffs.shape()[1];

  // Get the one_rdm that transforms from atomic orbitals to one_rdm using molecular orbitals and occupations.
  //  Here I"m using numpy to do the dot products, then it passes by value to C++, where then I obtain it.
  auto local = py::dict();
  local["iodata_obj"] = iodata_obj;
  local["mo_one_rdm"];
  py::exec(R"(
        # Convert to the Gaussian (.fchk) format
        import numpy as np
        from iodata.convert import convert_conventions, HORTON2_CONVENTIONS

        conventions = HORTON2_CONVENTIONS
        conventions[(0, 'c')] = ['1']
        conventions[(1, 'c')] = ['x', 'y', 'z']
        conventions[(2, 'c')] = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        conventions[(3, 'c')] = ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
        conventions[(4, 'c')] = ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy',
                                  'xxzz', 'xxyz', 'xxyy', 'xxxz', 'xxxy', 'xxxx']
        conventions[(5, 'c')] = HORTON2_CONVENTIONS[(5, 'c')][::-1]
        conventions[(6, 'c')] = HORTON2_CONVENTIONS[(6, 'c')][::-1]
        conventions[(7, 'c')] = HORTON2_CONVENTIONS[(7, 'c')][::-1]
        conventions[(8, 'c')] = HORTON2_CONVENTIONS[(8, 'c')][::-1]
        conventions[(9, 'c')] = HORTON2_CONVENTIONS[(9, 'c')][::-1]

        permutation, signs = convert_conventions(iodata_obj.obasis, conventions)
        coeffs = signs[:, np.newaxis] * iodata_obj.mo.coeffs[permutation]
        mo_one_rdm = (coeffs * iodata_obj.mo.occs).dot(coeffs.T)
    )", py::globals(), local);
  py::array_t<double, py::array::c_style> mo_one_rdm = local["mo_one_rdm"].cast<py::array_t<double>>();

  // Get the coordinates of the atoms.
  int natoms = iodata_obj.attr("natom").cast<int>();
  py::array_t<double, py::array::c_style> coords_atoms = iodata_obj.attr("atcoords").cast<py::array_t<double>>();

  // Get Charges of the atoms.
  long int* charges = new long int[natoms];
  py::array_t<long int> charges_atoms = iodata_obj.attr("atcorenums").cast<py::array_t<long int>>();
  for(int i = 0; i < natoms; i++) {charges[i] = charges_atoms.at(i);}

  // Get Atomic Numbers of the atoms
  long int* atnums = new long int[natoms];
  py::array_t<long int> atnums_atoms = iodata_obj.attr("atnums").cast<py::array_t<long int>>();
  for(int i = 0; i < natoms; i++) {atnums[i] = atnums_atoms.at(i);}

  // Commit them to memory so that pybind11 doesn't wipe them out. This is done using column order.
  double* h_coeffs = new double[one_rdm_shape_row * one_rdm_shape_col];
  double* h_occs = new double[one_rdm_shape_col];
  double* h_coords_atoms = new double[natoms * 3];
  double* h_one_rdm = new double[one_rdm_shape_row * one_rdm_shape_row];
  double* h_mo_one_rdm = new double[one_rdm_shape_row * one_rdm_shape_row];
  for(int i = 0; i < one_rdm_shape_row; i++){
    // Iterate coefficients in row-major order
    for(int k = 0; k < one_rdm_shape_col; k++) {
      h_coeffs[k + i * one_rdm_shape_col] = coeffs.at(i, k);
    }
    // Iterate the column. Update h_mo_one_rdm in column-major order.
    for(int j = 0; j < one_rdm_shape_row; j++){
      // Update one rdm if it isn't empty
      h_mo_one_rdm[i + j * one_rdm_shape_row] = mo_one_rdm.at(i, j);
      // Update scf one -rdm
      if (! one_rdms.empty()) {
        h_one_rdm[i + j * one_rdm_shape_row] = one_rdm.at(i, j);
      }
      else {
        h_one_rdm[i + j * one_rdm_shape_row] = 0.0;
      }
    }
  }
  for(int i = 0; i < one_rdm_shape_col; i++) {
    h_occs[i] = occs.at(i);
  }
  for(int i = 0; i < natoms; i++) {
    for(int k = 0; k < 3; k++) {
      h_coords_atoms[k + i * 3] = coords_atoms.at(i, k);
    }
  }
  // Finalize the interpreter.
  return {chemtools::MolecularBasis(molecular_basis), h_coords_atoms, natoms,
          h_one_rdm, {one_rdm_shape_row, one_rdm_shape_col}, h_coeffs, h_occs, charges, atnums, h_mo_one_rdm};
}


chemtools::IOData::IOData(const chemtools::IOData& copy):
  natoms(copy.natoms),
  one_rdm_shape_(copy.one_rdm_shape_)
{
  int nbasis = copy.orbital_basis_.numb_basis_functions();
  orbital_basis_ = chemtools::MolecularBasis(copy.orbital_basis_);
  charges_ = new long int[natoms];
  std::memcpy(charges_, copy.charges_, sizeof(long int) * natoms);
  atnums_ = new long int[natoms];
  std::memcpy(atnums_, copy.atnums_, sizeof(long int) * natoms);
  coord_atoms_ = new double[3 * natoms];
  std::memcpy(coord_atoms_, copy.coord_atoms_, sizeof(double) * 3 * natoms);
  one_rdm_ = new double[copy.one_rdm_shape_[0] * copy.one_rdm_shape_[0]];
  std::memcpy(one_rdm_, copy.one_rdm_, sizeof(double) * copy.one_rdm_shape_[0] * copy.one_rdm_shape_[0]);
  mo_coeffs_ = new double[copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]];
  std::memcpy(mo_coeffs_, copy.mo_coeffs_, sizeof(double) * copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]);
  mo_occupations_ = new double[one_rdm_shape_[1]];
  std::memcpy(mo_occupations_, copy.mo_occupations_, sizeof(double) * one_rdm_shape_[1]);
  mo_one_rdm_ = new double[one_rdm_shape_[0] * one_rdm_shape_[0]];
  std::memcpy(mo_one_rdm_, copy.mo_one_rdm_, sizeof(double) * one_rdm_shape_[0] * one_rdm_shape_[0]);
}
