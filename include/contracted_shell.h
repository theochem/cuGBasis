#ifndef CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
#define CHEMTOOLS_CUDA_CONTRACTED_SHELL_H

#include <array>
#include <assert.h>
#include <vector>

namespace chemtools {

inline int get_number_of_basis_functions_in_shell(int angmom) {
  if (angmom == 0) {
    return 1;
  }
  else if (angmom == 1) {
    return 3;
  }
  else if (angmom == 2) {
    return 6;
  }
  else if (angmom == -2) {
    // (-2) means it is pure type D-type shell
    return 5;
  }
  else if (angmom == 3) {
    return 10;
  }
  else if (angmom == -3) {
    // (-3) means it is pure type F-type shell
    return 7;
  }
  else if (angmom == 4) {
    return 15;
  }
  else if (angmom == -4) {
    return 9;
  }
}

struct GeneralizedContractionShell {
  std::vector<int>                 angmoms;     // Angular momentum are either 0, 1, 2, -2, 3, -3 (Gaussian FHCK)
  std::array<double, 3>            coordinate;
  std::vector<double>              exponents;
  std::vector<std::vector<double>> coefficients;
};

class MolecularBasis {
 public:
  std::vector<GeneralizedContractionShell> shells;

  MolecularBasis() {
    shells = {GeneralizedContractionShell{{0}, {0, 0, 0}, {0}, {{0}}}};
  }
  MolecularBasis(std::vector<GeneralizedContractionShell> shells) {this->shells = shells;}
  // Number of contracted shells.
  int numb_contracted_shells(bool decontracted = false) const {
    if (decontracted) {
       int counter = 0;
       for(GeneralizedContractionShell contr_shell : shells) {
         counter += contr_shell.angmoms.size();
       }
       return counter;
    }
    return shells.size();
  };
  MolecularBasis(const MolecularBasis& copy) : shells(copy.shells) {};

  // Number of contractions/atomic orbitals.
  int numb_basis_functions() const {
    int numb = 0;
    for(GeneralizedContractionShell contr_shell : shells) {
      for(int angmom : contr_shell.angmoms) {
        numb += get_number_of_basis_functions_in_shell(angmom);
      }
    }
    return numb;
  }
};
} // end CHEMTOOLS

#endif //CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
