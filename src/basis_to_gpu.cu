#include <vector>
#include <cassert>

#include "../include/contracted_shell.h"
#include "../include/basis_to_gpu.cuh"

// Stores constant memory for the NVIDIA GPU.
__constant__ double g_constant_basis[7500];


__host__ void gbasis::add_mol_basis_to_constant_memory_array(
    const gbasis::MolecularBasis& basis, bool do_segmented_basis, const bool disp) {
  /**
   * The following explains how memory is placed is with constant memory.
   * 4 byte blank means the previous number was a integer (since its 4 bytes and this is a double array).
   * M := numb_segmented_shell inside contracted shell.
   * K := Number of primitives per Contraction in Mth segmented shell (ie number of exponents).
   *
   * Normal mode:
   *  | numb_contracted_shells | 4 byte blank | atom_coord_x | atom_coord_y | atom_coord_z | M | 4B Blank | K | ...
   *     4B Blank | exponent_1 | ... | exponent_K | angmom_1 | coeff_11 | ... | coeff_K1 | ... | angmom_M | coeff_1M |..
   *     coeff_KM | K2 | 4B Blank | exponent_2 | same pattern repeat...
   * Segmented mode:
   *  | numb_contracted_shells | 4 byte blank | atom_coord_x | atom_coord_y | atom_coord_z | K | ...
   *     4B Blank | angmom_1 | exponent_1 | ... | exponent_K |  coeff_1 | ... | coeff_K | atom_coord_x | atom_coord_y |
   *     atom_coord_z |  K2 | 4B Blank |  angmom_2 | exponent_1 | ... | exponent_K2 | coeff_1 | same pattern repeat...
   *
   *  Decontracted mode takes up more memory and may be unsuitable in some cases.
   */
  int numb_contracted_shells = basis.numb_contracted_shells(do_segmented_basis);
  if (disp) {
    printf("Number of contrated shells host %d \n", numb_contracted_shells);
  }

  // Figure out the total amount of double you need.
  std::vector<double> h_information;
  h_information.push_back((double) numb_contracted_shells);
  if (fabs(h_information[0]) < 1e-8){
    throw std::runtime_error("Number of contracted shell is zero.");
  }
  for(gbasis::GeneralizedContractionShell shell : basis.shells) {
    if (do_segmented_basis){
      // Go through segmented shell.
      for(int i = 0; i < shell.angmoms.size(); i++) {
        // Add atom coordinate
        h_information.push_back(shell.coordinate[0]);
        h_information.push_back(shell.coordinate[1]);
        h_information.push_back(shell.coordinate[2]);
        // Add number of primitives of this shell.
        h_information.push_back((double) shell.exponents.size());
        // Add angular momentum of segmented shell.
        h_information.push_back((double) shell.angmoms[i]);
        // Add all exponents.
        for(double exponent : shell.exponents) {h_information.push_back(exponent);}
        // Add coefficients of the segmented shell/
        for(double coeff : shell.coefficients[i]) {h_information.push_back(coeff);}
      }
    }
    else {
      // Add atom coordinate
      h_information.push_back(shell.coordinate[0]);
      h_information.push_back(shell.coordinate[1]);
      h_information.push_back(shell.coordinate[2]);
      // Add number of segmented shells in contracted shell.
      h_information.push_back(shell.angmoms.size());
      if (disp) {
        printf("Number of segmented shell %zu \n", shell.angmoms.size());
      }

      // Add number of primitives of this shell.
      h_information.push_back((double) shell.exponents.size());
      // Add all exponents.
      for(double exponent : shell.exponents) {
        h_information.push_back(exponent);
      }
      // Go through segmented shell.
      for(int i = 0; i < shell.angmoms.size(); i++) {
        // Add angular momentum of segmented shell.
        h_information.push_back((double) shell.angmoms[i]);
        if (disp) {
          printf("Ith sehll %d   and angmom %f \n ", i, h_information[h_information.size() - 1]);
        }
        // Add coefficients of the segmented shell/
        for(double coeff : shell.coefficients[i]) {
          h_information.push_back(coeff);
        }
      }
    }
  }

  if (disp) {
    printf("Basis set information put into constant memory : \n");
    for (double a : h_information) {
      printf("  %f  ", a);
    }
    printf("\n");
  }

  std::size_t number_of_bytes = sizeof(double) * h_information.size();
  if (disp) {
    printf("Host: Constant memory number of bytes %zu \n", number_of_bytes);
  }
  if (number_of_bytes > 64000) {
    throw std::runtime_error("The basis set exceeds constant memory size. "
                             "Making it work on portions of hte basis set that fits "
                             "the size hasn't been implemented yet");
  }
  // Copy that array over to constant memory.
  double* h_info = h_information.data();

//  printf("\n Copy information over to constant memory. \n");
  cudaError_t _m_cudaStat = cudaMemcpyToSymbol(g_constant_basis, h_info, number_of_bytes, 0, cudaMemcpyHostToDevice);
  if (_m_cudaStat != cudaSuccess) {
    fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);
    throw std::runtime_error("Copying to constant memory did not work.");
  }
//  printf(" Stoping \n ");
}


__host__ void gbasis::add_mol_basis_to_constant_memory_array_access(gbasis::MolecularBasis basis) {
    /**
     * Basis set information is stored in constant memory as arrays.
     */

}
