/**
 * @file Responsible for wrapping around IOData.
 */
#ifndef CHEMTOOLS_CUDA_INCLUDE_IODATA_H_
#define CHEMTOOLS_CUDA_INCLUDE_IODATA_H_

#include <pybind11/pybind11.h>
#include <string>

#include "contracted_shell.h"

namespace py = pybind11;

namespace chemtools {
/**
 * IOData object that holds the atomic orbital, molecular obitals and one_rdm information.
 */
class IOData {
 private:
  chemtools::MolecularBasis orbital_basis_;
  double* coord_atoms_;      // Row-order
  int natoms;
  int* charges_;             // Charge of atom.
  double* one_rdm_;         // Row-order
  int one_rdm_shape_;
  double* mo_coeffs_;       // Row-order
  double* mo_occupations_;
  // Row-order, this is (mo_coeffs_ * mo_occupations).dot(mo_coeffs_). Used to evaluate Electron Density.
  double* mo_one_rdm_;


 public:
  IOData(chemtools::MolecularBasis basis, double* coord, int natoms,
         double* one_rdm, int shape, double* coeffs, double* occs, int* charges,
         double* mo_one_rdm) :
      orbital_basis_(basis), coord_atoms_(coord), natoms(natoms), one_rdm_(one_rdm), one_rdm_shape_(shape),
      mo_coeffs_(coeffs), mo_occupations_(occs), charges_(charges), mo_one_rdm_(mo_one_rdm) {}
  ~IOData() {
    delete one_rdm_; delete mo_coeffs_; delete mo_occupations_; delete coord_atoms_; delete charges_;
             delete mo_one_rdm_;
  }
  IOData(const IOData& copy);

  inline void print_molecular_coefficients(){
    int ncols = orbital_basis_.numb_basis_functions();
    for(int i = 0; i < one_rdm_shape_; i++) {
      for(int j = 0; j < ncols; j++){
        printf(" %f ", mo_coeffs_[j + i * ncols]);
      }
      printf("\n");
    }
  }

  inline void print_one_rdm(){
    for(int i = 0; i < one_rdm_shape_; i++) {
      for(int j = 0; j < one_rdm_shape_; j++){
        printf(" %f ", one_rdm_[j + i * one_rdm_shape_]);
      }
      printf("\n");
    }
    for(int i = 0; i < 5; i++) {
      printf(" Hello %.25f ", one_rdm_[i]);
    }
  }

  inline void print_coordinates(){
    for(int i = 0; i < natoms; i++){
      for(int j = 0; j < 3; j++) {
        printf(" %f ", coord_atoms_[j + i * 3]);
      }
      printf("\n");
    }
  }

  // Getters
  const MolecularBasis &GetOrbitalBasis() const {return orbital_basis_;}
  const double *GetCoordAtoms() const {return coord_atoms_;}
  int GetNatoms() const {return natoms;}
  const double *GetOneRdm() const {return one_rdm_;}
  const double *GetMOOneRDM() const {return mo_one_rdm_;}
  int GetOneRdmShape() const {return one_rdm_shape_;}
  const double *GetMoCoeffs() const {return mo_coeffs_;}
  const double *GetMoOccupations() const {return mo_occupations_;}
  const int*  GetCharges() const {return charges_;}

  // Setters
  //void SetCoordAtoms(double *coord_atoms) {IOData::coord_atoms_ = coord_atoms;}
};

/**
 * Obtain molecular basis from a format checkpoint file.
 *
 * Shell types (NShell values): 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
 *
 * @param fchk_file_path File path to the fchk file.
 * @param disp True then print the output.
 * @return MolecularBasis object holding the molecular basis set information.
 */
IOData get_molecular_basis_from_fchk(const std::string& fchk_file_path, bool disp=false);
} // end chemtools

#endif //CHEMTOOLS_CUDA_INCLUDE_IODATA_H_
