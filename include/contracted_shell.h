#ifndef CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
#define CHEMTOOLS_CUDA_CONTRACTED_SHELL_H

#include <array>
#include <assert.h>
#include <vector>

namespace chemtools {

/**
 * @brief Calculates the number of basis functions in a shell based on angular momentum
 * @param angmom Angular momentum value (positive for Cartesian, negative for pure/spherical)
 * @return Number of basis functions
 */
constexpr int get_number_of_basis_functions_in_shell(int angmom) {
    switch (std::abs(angmom)) {
        case 0: return 1;  // S-type
        case 1: return 3;  // P-type
        case 2: return (angmom < 0) ? 5 : 6;   // D-type (pure/Cartesian)
        case 3: return (angmom < 0) ? 7 : 10;  // F-type (pure/Cartesian)
        case 4: return (angmom < 0) ? 9 : 15;  // G-type (pure/Cartesian)
        default:
            assert(false && "Unsupported angular momentum");
            return 0;
    }
}


/**
 * @brief Represents a Generalized Contraction Shell
 */
struct GeneralizedContractionShell {
  std::vector<int>                 angmoms;
  std::array<double, 3>            coordinate;
  std::vector<double>              exponents;
  std::vector<std::vector<double>> coefficients;
};


/**
 * @brief Represents a molecular basis as a collection of contraction shells
 *        Similar to GBasis
 */
class MolecularBasis {
public:
    // Constructors
    MolecularBasis()
        : shells{GeneralizedContractionShell{{0}, {0.0, 0.0, 0.0}, {0.0}, {{0.0}}}} {}
    explicit MolecularBasis(std::vector<GeneralizedContractionShell> shells_)
        : shells(std::move(shells_)) {}
    MolecularBasis(const MolecularBasis&) = default;
    MolecularBasis& operator=(const MolecularBasis&) = default;
    MolecularBasis(MolecularBasis&&) noexcept = default;
    MolecularBasis& operator=(MolecularBasis&&) noexcept = default;
    
    std::vector<GeneralizedContractionShell> shells;
    
    
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
} // namespace chemtools

#endif //CHEMTOOLS_CUDA_CONTRACTED_SHELL_H
