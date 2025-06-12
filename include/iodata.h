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
  MolecularBasis      orbital_basis_;        // Basis-Set
  double*             coord_atoms_;          // Row-order
  int                 natoms;
  long int*           charges_;              // Charge of atom.
  long int*           atnums_;               // Atomic Numbers of atom
  double*             one_rdm_;              // Row-order
  std::array<int, 2>  one_rdm_shape_;
  /// Column-order is done to make it easy to do matrix-matrix multiplcation with Transpose C^T
  double*             mo_coeffs_col_;            // Col-order AO->MO Transformation
  double*             mo_coeffs_col_a_;          // Col-order AO->MO (Spin-Alpha) Transformation
  double*             mo_coeffs_col_b_;          // Col-order AO->MO (Spin-Beta) Transformation
  std::array<int, 2>  mo_coeffs_col_shape_;
  double*             mo_occupations_;
  double*             mo_occs_a_;            // Orbital occupation of alpha electrons.
  double*             mo_occs_b_;            // Orbital occupation of beta electrons.
  double*             mo_one_rdm_;           // Both Spins One-RDM matrix (symmetric).
  double*             mo_one_rdm_a_;         // Spin-Alpha One-RDM matrix (symmetric).
  double*             mo_one_rdm_b_;         // Spin-Beta One-RDM matrix (symmetric).

 public:
  IOData(
    MolecularBasis basis,
    double* coord,
    int natoms,
    double* one_rdm,
    std::array<int, 2> shape,
    double* coeffs,
    double* coeffs_a,
    double* coeffs_b,
    std::array<int, 2> mo_coeffs_col_shape,
    double* occs,
    double* occs_a,
    double* occs_b,
    long int* charges,
    long int* atnums,
    double* mo_one_rdm,
    double* mo_one_rdm_a,
    double* mo_one_rdm_b
    ) :
      orbital_basis_(basis),
      coord_atoms_(coord),
      natoms(natoms),
      one_rdm_(one_rdm),
      one_rdm_shape_(shape),
      mo_coeffs_col_(coeffs),
      mo_coeffs_col_a_(coeffs_a),
      mo_coeffs_col_b_(coeffs_b),
      mo_coeffs_col_shape_(mo_coeffs_col_shape),
      mo_occupations_(occs),
      mo_occs_a_(occs_a),
      mo_occs_b_(occs_b),
      charges_(charges),
      atnums_(atnums),
      mo_one_rdm_(mo_one_rdm),
      mo_one_rdm_a_(mo_one_rdm_a),
      mo_one_rdm_b_(mo_one_rdm_b) {}
  ~IOData() {
    delete one_rdm_; delete mo_coeffs_col_; delete mo_coeffs_col_a_; delete mo_coeffs_col_b_;
    delete mo_occupations_; delete coord_atoms_; delete charges_; delete mo_occs_a_; delete mo_occs_b_;
    delete atnums_; delete mo_one_rdm_; delete mo_one_rdm_a_; delete mo_one_rdm_b_;
  }

  IOData(const IOData& copy):
    natoms(copy.natoms), one_rdm_shape_(copy.one_rdm_shape_), mo_coeffs_col_shape_(copy.mo_coeffs_col_shape_)
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
    //TODO: Bug here because the shape of mo_coeffs_col_ is not correct it is (M, 2M)
    mo_coeffs_col_ = new double[copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]];
    std::memcpy(mo_coeffs_col_, copy.mo_coeffs_col_, sizeof(double) * copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]);
    mo_coeffs_col_a_ = new double[copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]];
    std::memcpy(mo_coeffs_col_a_, copy.mo_coeffs_col_a_, sizeof(double) * copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]);
    mo_coeffs_col_b_ = new double[copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]];
    std::memcpy(mo_coeffs_col_b_, copy.mo_coeffs_col_b_, sizeof(double) * copy.one_rdm_shape_[0] * copy.one_rdm_shape_[1]);
    mo_occupations_ = new double[one_rdm_shape_[1]];
    std::memcpy(mo_occupations_, copy.mo_occupations_, sizeof(double) * one_rdm_shape_[1]);

    mo_occs_a_ = new double[mo_coeffs_col_shape_[1]];
    std::memcpy(mo_occs_a_, copy.mo_occs_a_, sizeof(double) * mo_coeffs_col_shape_[1]);
    mo_occs_b_ = new double[mo_coeffs_col_shape_[1]];
    std::memcpy(mo_occs_b_, copy.mo_occs_b_, sizeof(double) * mo_coeffs_col_shape_[1]);

    mo_one_rdm_ = new double[one_rdm_shape_[0] * one_rdm_shape_[0]];
    std::memcpy(mo_one_rdm_, copy.mo_one_rdm_, sizeof(double) * one_rdm_shape_[0] * one_rdm_shape_[0]);
    mo_one_rdm_a_ = new double[one_rdm_shape_[0] * one_rdm_shape_[0]];
    std::memcpy(mo_one_rdm_a_, copy.mo_one_rdm_a_, sizeof(double) * one_rdm_shape_[0] * one_rdm_shape_[0]);
    mo_one_rdm_b_ = new double[one_rdm_shape_[0] * one_rdm_shape_[0]];
    std::memcpy(mo_one_rdm_b_, copy.mo_one_rdm_b_, sizeof(double) * one_rdm_shape_[0] * one_rdm_shape_[0]);
  }

  inline void print_molecular_coefficients(){
    // mo coefficients is in row-major order
    for(int i = 0; i < one_rdm_shape_[0]; i++) {
      for(int j = 0; j < one_rdm_shape_[1]; j++){
        printf(" %f ", mo_coeffs_col_[j + i * one_rdm_shape_[1]]);
      }
      printf("\n");
    }
  }

  inline void print_one_rdm(){
    for(int i = 0; i < one_rdm_shape_[0]; i++) {
      for(int j = 0; j < one_rdm_shape_[0]; j++){
        printf(" %f ", one_rdm_[j + i * one_rdm_shape_[0]]);
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
  double *GetCoordAtoms() const {return coord_atoms_;}
  int GetNatoms() const {return natoms;}

  /// One-RDM of different spin-types
  const double *GetOneRdm(std::string type ) const {return one_rdm_;}
  const double *GetMOOneRDM(std::string type = "ab") const {
    if (type == "ab") {
      return mo_one_rdm_;
    }
    if (type == "a") {
      return mo_one_rdm_a_;
    }
    if (type == "b") {
      return mo_one_rdm_b_;
    }
    throw std::runtime_error("IOData::GetMOOneRDM: Unknown type");
  }
  int GetOneRdmShape() const {return one_rdm_shape_[0];}
  const double *GetMOCoeffs(std::string type = "ab") const {
    if (type == "ab")
      return mo_coeffs_col_;
    if (type == "a")
      return mo_coeffs_col_a_;
    if (type == "b")
      return mo_coeffs_col_b_;
    throw std::runtime_error("IOData::GetOneRdm: Unknown type");
  }
  int GetMOCoeffsRow() const {return mo_coeffs_col_shape_[0];}
  int GetMOCoeffsCol() const {return mo_coeffs_col_shape_[1];}
  const double *GetMoOccupations() const {return mo_occupations_;}
  const double *GetMoAlphaOccupations() const {return mo_occs_a_;}
  const double *GetMoBetaOccupations() const {return mo_occs_b_;}
  const long int*  GetCharges() const {return charges_;}
  const long int* GetAtomicNumbers() const {return atnums_;}

  // Setters
  //void SetCoordAtoms(double *coord_atoms) {IOData::coord_atoms_ = coord_atoms;}

  int GetHOMOIndex(std::string type = "a") {
    // Returns the (0-based) index for HOMO orbitals
    int index = one_rdm_shape_[1] - 1;
    if (type == "a") {
      for (int i = 0; i < one_rdm_shape_[1]; i++) {
        if (mo_occs_a_[i] < 1E-6) {
          index = i - 1;
          break;
        }
      }
    }
    else if (type == "b") {
      for (int i = 0; i < one_rdm_shape_[1]; i++) {
        if (mo_occs_b_[i] < 1E-6) {
          index = i - 1;
          break;
        }
      }
    }
    else {
      throw std::runtime_error("IOData::GetOneRdm: Unknown spin type");
    }
    return index;
  }
  int GetLUMOIndex(std::string type = "a") {
    // Returns the (0-based) index for LUMO orbitals
    int index = -1;
    if (type == "a") {
      for (int i = 0; i < this->GetMOCoeffsCol(); i++) {
        if (mo_occs_a_[i] < 1E-6) {
          index = i;
          break;
        }
      }
    }
    else if (type == "b") {
      for (int i = 0; i < this->GetMOCoeffsCol(); i++) {
        if (mo_occs_b_[i] < 1E-6) {
          index = i;
          break;
        }
      }
    }
    else {
      throw std::runtime_error("IOData::GetOneRdm: Unknown spin type");
    }
    if (index == -1) {
      throw std::runtime_error("IOData::GetLUMOIndex: LUMOIndex not found");
    }
    return index;
  }
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
