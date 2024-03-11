#include <vector>
#include <cassert>
#include <exception>
#include <stdexcept>

#include "../include/contracted_shell.h"
#include "../include/basis_to_gpu.cuh"
#include "../include/cuda_utils.cuh"


// Stores constant memory for the NVIDIA GPU. Note 7500 * 8 bytes = 60 KB
//  Leaving 4 KB of constant memory left
__constant__ double g_constant_basis[7500];


__host__ std::array<std::size_t, 2> chemtools::add_mol_basis_to_constant_memory_array(
    const chemtools::MolecularBasis& basis, bool do_segmented_basis, const bool disp, const std::size_t i_shell_start
    ) {
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
  // Also includes the segmented part
  int numb_contracted_shells = basis.numb_contracted_shells(do_segmented_basis);
  if (disp) {
    printf("Number of contrated shells host %d \n", numb_contracted_shells);
  }

  // Figure out the total amount of double you need.
  std::vector<double> h_information;
  h_information.push_back(0.0);  // Update Number of contracted shell (Gets Updated Later).

  // Turn this into a while-loop based on the number of bytes
  std::size_t number_of_bytes = sizeof(double) * h_information.size();
  std::size_t index = i_shell_start;
  bool continue_iteration = true;
  std::size_t number_contractions_in_constant_memory = 0;
  int numb_contracted_shells_no_segmented = basis.numb_contracted_shells(false);  // Groups them together.
  while((index < numb_contracted_shells_no_segmented) & continue_iteration) {
    if (disp) {
      printf("Index to do: %zu  < %d  number of bytes starting: %zu \n", index, numb_contracted_shells, number_of_bytes);
    }
    chemtools::GeneralizedContractionShell shell = basis.shells[index];

    std::vector<double> temp;
    std::size_t temp_contraction_shell = 0;
    if (do_segmented_basis){
      // Go through segmented shell.
      for(int i = 0; i < shell.angmoms.size(); i++) {
        if (disp) {
          printf("Add shell with angmom %d \n", shell.angmoms[i]);
        }
        // Add atom coordinate
        temp.push_back(shell.coordinate[0]);
        temp.push_back(shell.coordinate[1]);
        temp.push_back(shell.coordinate[2]);
        // Add number of primitives of this shell.
        temp.push_back((double) shell.exponents.size());
        // Add angular momentum of segmented shell.
        temp.push_back((double) shell.angmoms[i]);
        // Add number of basis-functions
        temp_contraction_shell += chemtools::get_number_of_basis_functions_in_shell(shell.angmoms[i]);
        // Add all exponents.
        for(double exponent : shell.exponents) {temp.push_back(exponent);}
        // Add coefficients of the segmented shell/
        for(double coeff : shell.coefficients[i]) {temp.push_back(coeff);}
        // Update the number of contracted shells
        h_information[0] += 1.0;
      }
    }
    else {
      // Add atom coordinate
      temp.push_back(shell.coordinate[0]);
      temp.push_back(shell.coordinate[1]);
      temp.push_back(shell.coordinate[2]);
      // Add number of segmented shells in contracted shell.
      temp.push_back((double) shell.angmoms.size());
      if (disp) {
        printf("Number of segmented shell %zu \n", shell.angmoms.size());
      }

      // Add number of primitives of this shell.
      temp.push_back((double) shell.exponents.size());
      // Add all exponents.
      for(double exponent : shell.exponents) {
        temp.push_back(exponent);
      }
      // Go through segmented shell.
      for(int i = 0; i < shell.angmoms.size(); i++) {
        // Add angular momentum of segmented shell.
        temp.push_back((double) shell.angmoms[i]);
        // Add number of basis-functions
        temp_contraction_shell += chemtools::get_number_of_basis_functions_in_shell(shell.angmoms[i]);
        if (disp) {
          printf("Ith sehll %d   and angmom %f \n ", i, temp[temp.size() - 1]);
        }
        // Add coefficients of the segmented shell/
        for(double coeff : shell.coefficients[i]) {
          temp.push_back(coeff);
        }
      }
    } // end if segmented or not

    // If adding this shell doesn't put us over the limit
    if (sizeof(double) * (h_information.size() + temp.size()) < 60000) {
      // Add temp to h_information
      for(double x : temp) {h_information.push_back(x);}

      // Move to the next shell and update number of bytes for the loop
      if (!do_segmented_basis) {
        h_information[0] += 1.0;
      }
      index += 1;
      number_of_bytes = sizeof(double) * h_information.size();
      // Update number of contractions
      number_contractions_in_constant_memory += temp_contraction_shell;
    }
    else {
      // Exit the while loop
      continue_iteration = false;
    }

  } // end while loop

//  if (disp) {
//    printf("Basis set information put into constant memory : \n");
//    for (double a : h_information) {
//      printf("  %f  ", a);
//    }
//    printf("\n");
//  }

  number_of_bytes = sizeof(double) * h_information.size();
  if (disp) {
    printf("Host: Constant memory number of bytes %zu \n", number_of_bytes);
  }
//  if (number_of_bytes > 60000) {
//    throw std::runtime_error("The basis set exceeds constant memory size. "
//                             "Making it work on portions of hte basis set that fits "
//                             "the size hasn't been implemented yet");
//  }

  // Copy that array over to constant memory.
  double* h_info = h_information.data();

  cudaError_t _m_cudaStat = cudaMemcpyToSymbol(g_constant_basis, h_info, number_of_bytes, 0, cudaMemcpyHostToDevice);
  if (_m_cudaStat != cudaSuccess) {
    chemtools::cuda_check_errors(_m_cudaStat);
    throw std::runtime_error("Copying to constant memory did not work.");
  }

  return {index, number_contractions_in_constant_memory};
}


__host__ void chemtools::add_mol_basis_to_constant_memory_array_access(chemtools::MolecularBasis basis) {
    /**
     * Basis set information is stored in constant memory as arrays.
     */

}

/// Template kernel that can assess the device-function pointer `func`.
__global__ void temp_kernel(
    d_func_t func, double* d_out, const double* d_pt, const int numb_pts, const int numb_cont, const int i_cont_start
    ) {
  (*func)(d_out, d_pt, numb_pts, numb_cont, i_cont_start);
}

__host__ void chemtools::evaluate_scalar_quantity(
    const chemtools::MolecularBasis& basis,
    bool do_segmented_basis,
    const bool disp,
    d_func_t func_eval,
    double* d_output_iter,
    const double* d_points_iter,
    const int knumb_points_iter,
    const int k_total_numb_contractions,
    dim3 threadsPerBlock,
    dim3 grid,
    cudaFuncCache l1_over_shared
) {
  // Set the tempory kernel to prefer L1 cache over shared memory.
  chemtools::cuda_check_errors(cudaFuncSetCacheConfig(temp_kernel, l1_over_shared));

  // Start at the first shell
  std::size_t i_shell = 0;                              // Controls which shells are in constant memory
  std::size_t numb_basis_funcs_completed = 0;           // Controls where to update the next set of contractions
  std::array<std::size_t, 2> i_shell_and_numb_basis{};  // Used to update how many shells and contractions are in
                                                        //    constant memory.
  // Increment through each shell that can fit inside constant memory.
  while(i_shell < basis.numb_contracted_shells()) {
    // Transfer the basis-set to constant memory
    i_shell_and_numb_basis = chemtools::add_mol_basis_to_constant_memory_array(
        basis, do_segmented_basis, disp, i_shell
        );

    // Evaluate the function over the GPU
    temp_kernel<<<grid, threadsPerBlock>>>(func_eval, d_output_iter, d_points_iter, knumb_points_iter, k_total_numb_contractions,
                                           static_cast<int>(numb_basis_funcs_completed));
    cudaDeviceSynchronize();

    // Update to the next begining of the shells and contractions
    i_shell = i_shell_and_numb_basis[0];
    numb_basis_funcs_completed += i_shell_and_numb_basis[1];
  }
}