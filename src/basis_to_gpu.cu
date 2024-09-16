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


__host__ std::array<std::size_t, 2> chemtools::add_promol_basis_to_constant_memory_array(
    const double *const atom_coords,
    const long int* const atom_numbers,
    int natoms,
    const std::unordered_map<std::string, std::vector<double>>& promol_coeffs,
    const std::unordered_map<std::string, std::vector<double>>& promol_exps,
    const std::size_t index_atom_coords,
    const std::size_t i_atom_start
) {
  // Figure out the total amount of double you need.
  std::vector<double> h_information;

  // First put in the promolecular coefficients and exponents but only do it if `index_atom_coords` is zero.
  std::size_t new_index_atom_coords = index_atom_coords;  // This is used to update index_atom_coords when it is zero
  if (index_atom_coords == 0){
    // These set of elements should match what it is pymolecule.cu file inside the constructor of Promolecule
    std::vector<std::string> elements = {"h", "c", "n", "o", "f", "p", "s", "cl"};

    // The length of these tells where the coefficients of the hydrogen should start.
    for(auto x: elements) {
      h_information.push_back((double) elements.size());  // Points to 8th index where hydrogen starts
    }

    int n_elements_counter = 1;  // Controls how to update h_formation
    for (const std::string& element : elements) {
//      printf("n_elemnnets %d \n", n_elements_counter);

      // Place S-type First
      std::vector<double> coeffs_s = promol_coeffs.find(element + "_coeffs_s")->second;
      std::vector<double> exps_s = promol_exps.find(element + "_exps_s")->second;
//      if(element == "c") {
//        for(double x: coeffs_s){
//          printf(" %f ", x);
//        }
//        printf("\n");
//        for(double x: exps_s){
//          printf(" %f ", x);
//        }
//        printf("\n\n");
//      }
      h_information.push_back((double) coeffs_s.size());  // Places the number of coefficients
      // iteratively push coefficient then exponent
      int n_stype = coeffs_s.size();
      for(std::size_t i_param = 0; i_param < n_stype; i_param++){
        h_information.push_back((double) coeffs_s[i_param]);
        h_information.push_back((double) exps_s[i_param]);
      }

      // Place P-type First
      std::vector<double> coeffs_p = promol_coeffs.find(element + "_coeffs_p")->second;
      std::vector<double> exps_p = promol_exps.find(element + "_exps_p")->second;
      h_information.push_back((double) coeffs_p.size());  // Places the number of coefficients
      // iteratively push coefficient then exponent
      int n_ptype = coeffs_p.size();
      for(std::size_t i_param = 0; i_param < n_ptype; i_param++){
        h_information.push_back((double) coeffs_p[i_param]);
        h_information.push_back((double) exps_p[i_param]);
      }

      // Update the `n_elements_counter`th element to point to the correct place
      //     To calculate the index, first sum the previous element index, now add the number of s-type and p-type
      //     parameters (multiply by two because coeff & exp) then add two because we have to store the number of
      //     coefficients/exponents for s-type and p-type.
//      printf("Number s-type %d and p-type %d \n ", n_stype, n_ptype);
//      printf("Number s-type %f, %d and p-type %d \n ", h_information[n_elements_counter - 1], 2 * n_stype, 2 * n_ptype);
//      printf("n_elemnnets %d \n", n_elements_counter);
      if (n_elements_counter != elements.size()){  // Skip the last element
        h_information[n_elements_counter] = h_information[n_elements_counter - 1] + 2.0 * n_stype + 2.0 * n_ptype + 2.0;
      }
//      printf("\n");

      // Move to the next element
      n_elements_counter += 1;
    }

    // Update `index_atom_coords` so that it points to place where constant memory should now give the atomic
    //  coordinates, plus one so it points to the next index
    new_index_atom_coords = h_information.size();
  }

  // Place the atomic coordinates and atomic numbers inside constant memory
  //    Only go up to the amount of information that you are able to.
  std::size_t curr_atom_index = i_atom_start;  // Start at the specified atom
  // Keep iterating through the next atomic number (size 1) and  coordinate (size 3) if it doesn't exceed
  //  constant memory
  //  Recall `index_atom_coords` is the index where the atomic coordinates sohuld be places and
  //  h_information.size() Should be non-zero only when index_atom_coords is zero
  //  Add 3 since atomic coordinate has three elements
  int index_number_atoms_chunk;
  if (index_atom_coords == 0) {
    index_number_atoms_chunk = new_index_atom_coords;
  }
  else {
    index_number_atoms_chunk = 0;  // The first element updates
  }
  h_information.push_back(0.0);  // This is suppose to hold the number of atoms inside constant memory.
//  printf("Index number atoms chunk %d \n ", index_number_atoms_chunk);
  while(index_atom_coords + h_information.size() + 4 <= 7500 & curr_atom_index < natoms) {
    // Instead of placing the atomic number, this instead places the index within constant memory
    //    where one would find the index where the promolecular coefficeints for that atom starts
    //    This should match the order within
    //    std::vector<std::string> elements = {"h", "c", "n", "o", "f", "p", "s", "cl"} line above
    //    For example, "c' would be the second index and so it should store two.
    int c = atom_numbers[curr_atom_index];
    if (c == 1) {// Hydrogen
      h_information.push_back(0);
    }
    else if (c == 6) { // Carbon
      h_information.push_back(1);
    }
    else if (c== 7) { // Nitrogen
      h_information.push_back(2);
    }
    else if (c == 8) {
      h_information.push_back(3);
    }
    else if (c == 9) {
      h_information.push_back(4);
    }
    else if (c == 15) { // Phosphorous
      h_information.push_back(5);
    }
    else if (c == 16) {  // Sulfur
      h_information.push_back(6);
    }
    else if ( c == 17) {
      h_information.push_back(7);
    }
    else {
      throw std::runtime_error("Could not recognize what atomic number it is provided, it isn'tprobably done " + std::to_string(c) + " \n");
    }
    //h_information.push_back(static_cast<double>());         // Atomic-Number
    h_information.push_back(atom_coords[curr_atom_index * 3]);      // X-Coordinate
    h_information.push_back(atom_coords[curr_atom_index * 3 + 1]);  // Y-Coordinate
    h_information.push_back(atom_coords[curr_atom_index * 3 + 2]);  // Z-Coordinate
    h_information[index_number_atoms_chunk] += 1;  // Update the number of atoms placed within constant memory.
    curr_atom_index += 1;
  }
  // At the end if curr_atom_index < natoms was the termination, then curr_atom_index == natoms
  // Thus you know you at completion when curr_atom_index == natoms.
//  for(int i = 0;i < h_information[0];i++){
//    printf(" %f ", h_information[i]);
//  }
//  printf("\n\n");
//  for(int i = h_information[1]; i < h_information[2] + 3; i++){
//    printf(" %f ", h_information[i]);
//  }
//  printf("\n\n");
//  for(int i = index_number_atoms_chunk; i < h_information.size(); i++){
//    printf(" %f ", h_information[i]);
//  }
//  printf("\n\n");
//
//  for(double x : h_information) {
//    printf(" %f ", x);
//  }
//  printf("\n index_number_atoms_chunk %f \n\n", h_information[new_index_atom_coords]);

  // Copy that array over to constant memory but if only from index_atom_coords to later
  double* h_info = h_information.data();
  cudaError_t _m_cudaStat = cudaMemcpyToSymbol(
      g_constant_basis,
      h_info,
      sizeof(double) * h_information.size(),
      sizeof(double) * index_atom_coords,  // Only update past index_atom_coords
      cudaMemcpyHostToDevice
      );
  if (_m_cudaStat != cudaSuccess) {
    chemtools::cuda_check_errors(_m_cudaStat);
    throw std::runtime_error("Copying to constant memory did not work.");
  }

  return {new_index_atom_coords, curr_atom_index};
}