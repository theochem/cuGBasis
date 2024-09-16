#include "../include/evaluate_promolecular.cuh"
#include "../include/basis_to_gpu.cuh"
#include "../include/cuda_utils.cuh"

#define CUDART_PI_D 3.141592653589793238462643383279502884197169

/**
 *
 * Device Functions
 *
 */
__global__ void chemtools::evaluate_promol_density_from_constant_memory_on_any_grid(
    double* d_density_array, const double* const d_points, const int knumb_points, const int index_atom_coords_start
) {
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index < knumb_points) {
//    if (global_index == 0) {
//      for(int i =0; i < index_atom_coords_start + 5; i++) {
//        printf(" %f  ", g_constant_basis[i]);
//      }
//    }
//    printf("Global index %u \n", global_index);
//    for(int i =0; i < 100; i++) {
//      printf(" %f  ", g_constant_basis[i]);
//    }
    int iconst = index_atom_coords_start;  // Index to go over the atomic coordinates
//    printf("iconst %d \n", iconst);
    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];
//    printf("Index %u  Pt %f   %f  %f \n", global_index, grid_x, grid_y, grid_z);

    // Evaluate the density value and store it in constant memory
//    printf("Number of atoms %f \n ", g_constant_basis[iconst]);
    int knatom = (int) g_constant_basis[iconst++]; // Get the number of atoms within constant mem
//    if(global_index == 0) {
//      printf("Number of atoms %d \n ", knatom);
//    }
    for(int i_atom = 0; i_atom < knatom; i_atom++) {
      // Get the Type of Atom and Atomic Coordinates for this atom
      int index_of_index_of_element = (int) g_constant_basis[iconst++];   // Should point to i^E, see basis_to_gpu.cuh
      double r_A_x = (grid_x - g_constant_basis[iconst++]);
      double r_A_y = (grid_y - g_constant_basis[iconst++]);
      double r_A_z = (grid_z - g_constant_basis[iconst++]);
//      printf("Index of index %d \n", index_of_index_of_element);

      // Gets the index where promolecular coefficient for this atom starts
      int index_of_promol_coeffs = (int) g_constant_basis[index_of_index_of_element];  // Should be i^E
//      printf("Index of Coeffcients %d \n", index_of_promol_coeffs);

      // Evaluate-S type Gaussians
//      printf("Number of s-type Gaussians %f \n", g_constant_basis[index_of_promol_coeffs]);

      int number_s_type_gaussians = (int) g_constant_basis[index_of_promol_coeffs++];
//      printf("Number of s-type Gaussians %d \n", number_s_type_gaussians);
      for(int i = 0; i < number_s_type_gaussians; i++) {
        double coeff = g_constant_basis[index_of_promol_coeffs++];  // Get Coefficient of Gaussian
        double exponent = g_constant_basis[index_of_promol_coeffs++];    // Get Exponent of Gaussian
//        double normalization = (exponent / CUDART_PI_D);
//        normalization = normalization * normalization * normalization;
//        normalization = sqrt(normalization);
        double normalization = pow(exponent / CUDART_PI_D, 1.5);
//        if((global_index == 0) & (i_atom == 0) ){
//          printf("Coeff Exponent  %f  %f  \n", coeff, exponent);
//        }
        d_density_array[global_index] += coeff * normalization * exp(-exponent * ( r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z));
      }

      // Evaluate P-type Gaussians
      int number_p_type_gaussians = (int) g_constant_basis[index_of_promol_coeffs++];
//      printf("Number of p-type Gaussians %d \n", number_p_type_gaussians);
      for(int i = 0; i < number_p_type_gaussians; i++) {
        double coeff = g_constant_basis[index_of_promol_coeffs++];  // Get Coefficient of Gaussian
        double exponent = g_constant_basis[index_of_promol_coeffs++];    // Get Exponent of Gaussian
        double r_sq =  ( r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z);
//        double normalization = (exponent * exponent / CUDART_PI_D);
//        normalization = normalization * normalization * normalization;
//        normalization = 2.0 * sqrt(normalization) / 3.0;
        double normalization = (2.0 * pow(exponent, 2.5)) / (3.0 * pow(CUDART_PI_D, 1.5));
//        printf("Coeff Exponent  %f  %f  \n", coeff, exponent);
        d_density_array[global_index] += coeff * normalization * r_sq * exp(-exponent * r_sq);
      }
    }
  }
}


__global__ void chemtools::evaluate_promol_electrostatic_from_constant_memory_on_any_grid(
    double* d_density_array, const double* const d_points, const int knumb_points, const int index_atom_coords_start
) {
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index < knumb_points) {
    int iconst = index_atom_coords_start;  // Index to go over the atomic coordinates

    // Get the grid points where `d_points` is in column-major order with shape (N, 3)
    double grid_x = d_points[global_index];
    double grid_y = d_points[global_index + knumb_points];
    double grid_z = d_points[global_index + knumb_points * 2];

    // Evaluate the density value and store it in constant memory
    int knatom = (int) g_constant_basis[iconst++]; // Get the number of atoms within constant mem
    for(int i_atom = 0; i_atom < knatom; i_atom++) {
      // Get the Type of Atom and Atomic Coordinates for this atom
      int index_of_index_of_element = (int) g_constant_basis[iconst++];   // Should point to i^E, see basis_to_gpu.cuh
      double r_A_x = (grid_x - g_constant_basis[iconst++]);
      double r_A_y = (grid_y - g_constant_basis[iconst++]);
      double r_A_z = (grid_z - g_constant_basis[iconst++]);

      // Gets the index where promolecular coefficient for this atom starts
      int index_of_promol_coeffs = (int) g_constant_basis[index_of_index_of_element];  // Should be i^E

      // Evaluate-S type Gaussians
      int number_s_type_gaussians = (int) g_constant_basis[index_of_promol_coeffs++];
      for(int i = 0; i < number_s_type_gaussians; i++) {
        double coeff = g_constant_basis[index_of_promol_coeffs++];  // Get Coefficient of Gaussian
        double exponent = g_constant_basis[index_of_promol_coeffs++];    // Get Exponent of Gaussian
        double r_sq =  sqrt(r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z);
        d_density_array[global_index] += coeff * erf(sqrt(exponent) * r_sq) / r_sq;
      }

      // Evaluate P-type Gaussians, derived by taking the derivative of exponent and Leibniz Rule
      int number_p_type_gaussians = (int) g_constant_basis[index_of_promol_coeffs++];
      for(int i = 0; i < number_p_type_gaussians; i++) {
        double coeff = g_constant_basis[index_of_promol_coeffs++];  // Get Coefficient of Gaussian
        double exponent = g_constant_basis[index_of_promol_coeffs++];    // Get Exponent of Gaussian
        double r_sq =  ( r_A_x * r_A_x + r_A_y * r_A_y + r_A_z * r_A_z);
        double r = sqrt(r_sq);
        double first_term = erf(sqrt(exponent) * r) / r;

        double exponential = exp(-exponent * r_sq);
        double normalization = (2.0 * pow(exponent, 2.5)) / (3.0 * pow(CUDART_PI_D, 1.5));
        d_density_array[global_index] += coeff * (
            first_term -
            normalization * CUDART_PI_D * exponential / (exponent * exponent )
            );
      }
    }
  }
}
/***
 *
 * Host Functions
 *
 */


__host__ std::vector<double> chemtools::evaluate_promol_scalar_property_on_any_grid(
    const double* const atom_coords,
    const long int *const atom_numbers,
    int natoms,
    const std::unordered_map<std::string, std::vector<double>>& promol_coeffs,
    const std::unordered_map<std::string, std::vector<double>>& promol_exps,
    const double* h_points,
    const int knumb_points,
    std::string property
) {
  chemtools::cuda_check_errors(cudaFuncSetCacheConfig(evaluate_promol_density_from_constant_memory_on_any_grid, cudaFuncCachePreferL1));

  // Output that is returned
  std::vector<double> hpromol_density(knumb_points);

  // Second step is to figure out the optimal number of points that fit in global memory.
  // For the density, the optimal number of points is solving 4N (3N for points, N for output)
  // This is solved by 11.5 GB = 4N * 8 bytes => N = 11.5 GB / (N * 8 Bytes)
  size_t t_numb_pts = knumb_points;
  size_t free_mem = 0;   // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  size_t t_numb_pts_of_each_chunk = (free_mem - 500000000) / (sizeof(double) * 4);
  //  printf(" Number of points each chunk %zu \n", t_numb_pts_of_each_chunk);

  // Iterate through each chunk of the points
  size_t index_to_copy = 0;  // Index on where to start copying to h_density (start of sub-grid)
  size_t i_iter = 0;
  while (index_to_copy < knumb_points) {
//    printf("Iter %zu \n", i_iter);
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
//    printf("Number of pts iter %zu \n", number_pts_iter);

    // Allocate device memory for output array, and set all elements to zero via cudaMemset.
    double *d_density;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_density, sizeof(double) * number_pts_iter));
    chemtools::cuda_check_errors(cudaMemset(d_density, 0, sizeof(double) * number_pts_iter));


    // Transfer optimal number of points to GPU memory, this is in column order with shape (N, 3)
    // Transfer grid points to GPU
    //  Becasue h_points is in column-order and we're slicing based on the number of points that can fit in memory.
    //  Need to slice each (x,y,z) coordinate seperately.
    double *d_points;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * number_pts_iter));
    for (int i_slice = 0; i_slice < 3; i_slice++) {
      chemtools::cuda_check_errors(cudaMemcpy(&d_points[i_slice * number_pts_iter],
                                              &h_points[i_slice * knumb_points + index_to_copy],
                                              sizeof(double) * number_pts_iter,
                                              cudaMemcpyHostToDevice));
    }

    // Evaluate the promolecular density over subset of atoms that fit within constant memory
    std::array<std::size_t, 2> constant_mem_info = {0, 0};
    // Iteration through until it is computed over all atoms
    while (constant_mem_info[1] < natoms) {
      // First step is to place the promolecular information and atomic coordinates inside constant memory.
      //  First index is index within constant memory where the atomic coordinates should start, needed for GPU code
      constant_mem_info = chemtools::add_promol_basis_to_constant_memory_array(
          atom_coords,
          atom_numbers,
          natoms,
          promol_coeffs,
          promol_exps,
          constant_mem_info[0],
          constant_mem_info[1]
      );
//      printf("Consta mem info %zu   %zu  \n", constant_mem_info[0], constant_mem_info[1]);

      dim3 threadsPerBlock(1024);
      dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
      if (property == "density") {
        // Evaluate Density
        chemtools::evaluate_promol_density_from_constant_memory_on_any_grid<<<grid, threadsPerBlock>>>(
            d_density, d_points, number_pts_iter, constant_mem_info[0]
        );
      }
      else if (property == "electrostatic") {
        // Evaluate electrostatic
        chemtools::evaluate_promol_electrostatic_from_constant_memory_on_any_grid<<<grid, threadsPerBlock>>>(
            d_density, d_points, number_pts_iter, constant_mem_info[0]
        );
      }
      else {
        throw std::runtime_error("Could not recognize what property it is: " + property);
      }
      cudaDeviceSynchronize();
    }
    // Free up the points
    cudaFree(d_points);

    // Transfer the density back to the CPU
    //    Since I'm computing a sub-grid at a time, need to update the index h_electron_density, accordingly.
    chemtools::cuda_check_errors(cudaMemcpy(&hpromol_density[0] + index_to_copy,
                                            d_density,
                                            sizeof(double) * number_pts_iter,
                                            cudaMemcpyDeviceToHost));

    // Free up the density values
    cudaFree(d_density);

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.
  }

  // Compute the point-charge
  if (property == "electrostatic") {
//    printf("Compute electrostatics \n");
    for(int i_pt = 0; i_pt < knumb_points; i_pt++){
      double pt_x = h_points[i_pt];
      double pt_y = h_points[knumb_points + i_pt];
      double pt_z = h_points[2 * knumb_points + i_pt];

      // Go through each atom and copute point-charge
      double pt_charge = 0.0;
      for(int i_atom = 0; i_atom < natoms; i_atom++) {
        double A_x = atom_coords[3 * i_atom];
        double A_y = atom_coords[3 * i_atom + 1];
        double A_z = atom_coords[3 * i_atom + 2];

        // Distance between point and atomic coordinate
        double distance = sqrt((pt_x - A_x) * (pt_x - A_x) + (pt_y - A_y) * (pt_y - A_y) + (pt_z - A_z) * (pt_z - A_z));
        pt_charge += static_cast<double>(atom_numbers[i_atom]) / distance;
      }

      // Add to final result
      hpromol_density[i_pt] = pt_charge - hpromol_density[i_pt];
    }
  }
  return hpromol_density;
}