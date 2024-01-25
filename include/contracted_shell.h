#ifndef CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
#define CHEMTOOLS_CUDA_CONTRACTED_SHELL_H

#include <array>
#include <assert.h>
#include <vector>

namespace chemtools {
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
      for (int i = 0; i < contr_shell.angmoms.size(); i++) {
        int angmom = contr_shell.angmoms[i];
        if (angmom == 0) {
          numb += 1;
        }
        else if (angmom == 1) {
          numb += 3;
        }
        else if (angmom == 2) {
          numb += 6;
        }
        else if (angmom == -2) {
          // (-2) means it is pure type D-type shell
          numb += 5;
        }
        else if (angmom == 3) {
          numb += 10;
        }
        else if (angmom == -3) {
          // (-3) means it is pure type F-type shell
          numb += 7;
        }
        else if (angmom == 4) {
          numb += 15;
        }
        else if (angmom == -4) {
          numb += 9;
        }
      }
    }
    return numb;
  }
};
} // end CHEMTOOLS

#endif //CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
