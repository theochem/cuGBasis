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
  // py::object molecular_orb_obj = iodata_obj.attr("mo");
  // py::array_t<double, py::array::c_style> mo_coeffs = (molecular_orb_obj.attr("coeffs")).cast<py::array_t<double>>();
  // py::array_t<double, py::array::c_style> occs = molecular_orb_obj.attr("occs").cast<py::array_t<double>>();

  // Get the one_rdm that transforms from atomic orbitals to one_rdm using molecular orbitals and occupations.
  //  Here I"m using numpy to do the dot products, then it passes by value to C++, where then I obtain it.
  auto local = py::dict();
  local["iodata_obj"] = iodata_obj;
  py::exec(R"(
        # Convert to the Gaussian (.fchk) format
        import numpy as np
        try:
            from iodata.convert import convert_conventions, HORTON2_CONVENTIONS
        except (ImportError, ModuleNotFoundError):
            from iodata.basis import convert_conventions, HORTON2_CONVENTIONS

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
        occs = iodata_obj.mo.occs
        mo_one_rdm = (coeffs * occs).dot(coeffs.T)

        occs_a, coeffs_a = iodata_obj.mo.occsa, signs[:, np.newaxis] * iodata_obj.mo.coeffsa[permutation]
        occs_b, coeffs_b = iodata_obj.mo.occsb, signs[:, np.newaxis] * iodata_obj.mo.coeffsb[permutation]
        mo_one_rdm_a = (coeffs_a * occs_a).dot(coeffs_a.T);
        mo_one_rdm_b = (coeffs_b * occs_b).dot(coeffs_b.T);
    )", py::globals(), local);
  // Store the One-RDM
  py::array_t<double, py::array::c_style> mo_one_rdm   = local["mo_one_rdm"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> mo_one_rdm_a = local["mo_one_rdm_a"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> mo_one_rdm_b = local["mo_one_rdm_b"].cast<py::array_t<double>>();

  // Store the transformation matrix form AO to MO
  py::array_t<double, py::array::c_style> occs = local["occs"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> occs_a = local["occs_a"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> occs_b = local["occs_b"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> mo_coeffs   = local["coeffs"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> mo_coeffs_a = local["coeffs_a"].cast<py::array_t<double>>();
  py::array_t<double, py::array::c_style> mo_coeffs_b = local["coeffs_b"].cast<py::array_t<double>>();
  int one_rdm_shape_row = mo_one_rdm.shape()[0];   // Symmetric square matrix
  int one_rdm_shape_col = mo_one_rdm.shape()[1];
  int mo_coefficients_col_a = mo_coeffs_a.shape()[1];
  int mo_coefficients_col_b = mo_coeffs_b.shape()[1];
  assert(mo_coefficients_col_a == mo_coefficients_col_b &&
    "Mismatch in number of molecular orbitals between alpha and beta spins");
  int mo_coefficients_row_a = mo_coeffs_a.shape()[0];

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

  // Assertion to check this
  assert(mo_coefficients_row_a == one_rdm_shape_row);
  // Commit them to memory so that pybind11 doesn't wipe them out. This is done using column order.
  auto* h_coeffs_col    = new double[one_rdm_shape_row * mo_coeffs.shape()[1]];
  auto* h_coeffs_col_a  = new double[mo_coefficients_row_a * mo_coefficients_col_a];
  auto* h_coeffs_col_b  = new double[mo_coefficients_row_a * mo_coefficients_col_a];
  auto* h_occs          = new double[mo_coeffs.shape()[1]];
  auto* h_occs_a        = new double[mo_coefficients_col_a];
  auto* h_occs_b        = new double[mo_coefficients_col_a];
  auto* h_coords_atoms  = new double[natoms * 3];
  auto* h_one_rdm       = new double[one_rdm_shape_row * one_rdm_shape_row];
  auto* h_mo_one_rdm    = new double[one_rdm_shape_row * one_rdm_shape_row];
  auto* h_mo_one_rdm_a  = new double[one_rdm_shape_row * one_rdm_shape_row];
  auto* h_mo_one_rdm_b  = new double[one_rdm_shape_row * one_rdm_shape_row];
  for(int i = 0; i < one_rdm_shape_row; i++){
    // Iterate coefficients in row-major order h_coeffs(row-major order)
    for (int k = 0; k < mo_coeffs.shape()[1]; k++) {
      h_coeffs_col[i * mo_coeffs.shape()[1] + k] = mo_coeffs.at(i, k);
    }
    // k goes over columns (h_coeffs_a is in col-major order)
    for(int k = 0; k < mo_coefficients_col_a; k++) {
      h_coeffs_col_a[k * mo_coefficients_row_a + i] = mo_coeffs_a.at(i, k);
      h_coeffs_col_b[k * mo_coefficients_row_a + i] = mo_coeffs_b.at(i, k);
    }

    // Iterate the column. Update h_mo_one_rdm in column-major order.
    for(int j = 0; j < one_rdm_shape_row; j++){
      // Update one rdm if it isn't empty
      h_mo_one_rdm[i + j * one_rdm_shape_row] = mo_one_rdm.at(i, j);
      h_mo_one_rdm_a[i + j * one_rdm_shape_row] = mo_one_rdm_a.at(i, j);
      h_mo_one_rdm_b[i + j * one_rdm_shape_row] = mo_one_rdm_b.at(i, j);

      // Update scf one -rdm
      if (! one_rdms.empty()) {
        h_one_rdm[i + j * one_rdm_shape_row] = one_rdm.at(i, j);
      }
      else {
        h_one_rdm[i + j * one_rdm_shape_row] = 0.0;
      }
    }
  }
  for(int i = 0; i < mo_coefficients_col_a; i++) {
    h_occs_a[i] = occs_a.at(i);
    h_occs_b[i] = occs_b.at(i);
  }
  for(int i = 0; i < mo_coeffs.shape()[1]; i++) {
    h_occs[i] = occs.at(i);
  }
  for(int i = 0; i < natoms; i++) {
    for(int k = 0; k < 3; k++) {
      h_coords_atoms[k + i * 3] = coords_atoms.at(i, k);
    }
  }

  // Finalize the interpreter.
  return {chemtools::MolecularBasis(molecular_basis), h_coords_atoms, natoms,
          h_one_rdm, {one_rdm_shape_row, one_rdm_shape_col}, h_coeffs_col, h_coeffs_col_a,
          h_coeffs_col_b, {mo_coefficients_row_a, mo_coefficients_col_a},
          h_occs, h_occs_a, h_occs_b, charges, atnums, h_mo_one_rdm,
    h_mo_one_rdm_a, h_mo_one_rdm_b};
}
