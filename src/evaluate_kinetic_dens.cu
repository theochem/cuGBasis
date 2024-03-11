
#include <thrust/device_vector.h>
#include "cublas_v2.h"

#include "../include/evaluate_kinetic_dens.cuh"
#include "../include/evaluate_gradient.cuh"
#include "../include/evaluate_laplacian.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/basis_to_gpu.cuh"


__host__ std::vector<double> chemtools::evaluate_pos_def_kinetic_density_on_any_grid_handle(
    cublasHandle_t &handle, chemtools::IOData &iodata, const double *h_points, const int knumb_points
) {
  // Set cache perference to L1
//  chemtools::cuda_check_errors(cudaFuncSetCacheConfig(
//      chemtools::evaluate_derivatives_contractions_from_constant_memory, cudaFuncCachePreferL1
//  ));

  // Get the function pointers to the correct GPU functions
  d_func_t h_deriv_contr_func;
  cudaMemcpyFromSymbol(&h_deriv_contr_func, chemtools::p_evaluate_deriv_contractions, sizeof(d_func_t));

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  const chemtools::MolecularBasis& molecular_basis = iodata.GetOrbitalBasis();
  //chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();

  // Electron density in global memory and create the handles for using cublas.
  std::vector<double> h_kinetic_dens(knumb_points);

  /**
   * Note that the maximum memory requirement is 3NM + M^2 + NM + M, where N=numb points and M=numb basis funcs.
   * Solving for 11Gb we have (3NM + M^2 + NM + M)8 bytes = 11Gb 1e9 (since 1e9 bytes = 1GB) Solve for N to get
   * N = (11 * 10^9 - M^2 - M) / (3M + M).  This is the optimal number of points that it can do.
   */
  size_t t_numb_pts = knumb_points;
  size_t t_nbasis = knbasisfuncs;
  size_t t_highest_number_of_bytes = sizeof(double) * (
      3 * t_numb_pts * t_nbasis + t_nbasis * t_numb_pts + t_nbasis * t_nbasis + t_nbasis
  );
  size_t free_mem = 0;  // in bytes
  size_t total_mem = 0;  // in bytes
  cudaError_t error_id = cudaMemGetInfo(&free_mem, &total_mem);
  //  printf("Total Free Memory Avaiable in GPU is %zu \n", free_mem);
  // Calculate how much memory can fit inside the GPU memory.
  size_t t_numb_chunks = t_highest_number_of_bytes / free_mem;
  // Calculate how many points we can compute with free memory minus 0.5 gigabyte for safe measures:
  //    This is calculated by solving (5 * N M + M^2 + 3N) * 8 bytes = Free memory (in bytes)  for N to get:
  size_t t_numb_pts_of_each_chunk = (
      ((free_mem - 500000000) / (sizeof(double)))  - t_nbasis * t_nbasis - t_nbasis) / (3 * t_nbasis + t_nbasis);
  if (t_numb_pts_of_each_chunk == 0 and t_numb_chunks > 1.0) {
    // Haven't handle this case yet
    assert(0);
  }

  // Transfer one-rdm from host/cpu memory to device/gpu memory.
  double *d_one_rdm;
  chemtools::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
  chemtools::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                              sizeof(double), iodata.GetMOOneRDM(),
                                              iodata.GetOneRdmShape(), d_one_rdm, iodata.GetOneRdmShape()));

  // Iterate through each chunk of the data set.
  size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
  size_t i_iter = 0;
  while(index_to_copy < knumb_points) {
    // For each iteration, calculate number of points it should do, number of bytes it corresponds to.
    // At the last chunk,need to do the remaining number of points, hence a minimum is used here.
    size_t number_pts_iter = std::min(
        t_numb_pts - i_iter * t_numb_pts_of_each_chunk, t_numb_pts_of_each_chunk
    );
    //    printf("Number of points %zu \n", number_pts_iter);
    //    printf("Maximal number of points in x-axis to do each chunk %zu \n", number_pts_iter);

    // Transfer grid points to GPU, this is in column order with shape (N, 3)
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

    // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
    double *d_deriv_contractions;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions,
                                         sizeof(double) * 3 * number_pts_iter * knbasisfuncs));
    dim3 threadsPerBlock(128);
    dim3 grid((number_pts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
//    chemtools::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
//        d_deriv_contractions, d_points, number_pts_iter, knbasisfuncs
//    );
    chemtools::evaluate_scalar_quantity(
        molecular_basis,
        false,
        false,
        h_deriv_contr_func,
        d_deriv_contractions,
        d_points,
        number_pts_iter,
        knbasisfuncs,
        threadsPerBlock,
        grid
    );

    // Free up points memory in device/gpu memory.
    cudaFree(d_points);


    // Allocate memory to hold the matrix-multiplcation between d_one_rdm and each `i`th derivative (i_deriv, M, N)
    double *d_temp_rdm_derivs;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs, sizeof(double) * number_pts_iter * knbasisfuncs));
    // Allocate device memory for the derivative of the one-electorn rdm column-major order.
    double *d_deriv_rdm;
    chemtools::cuda_check_errors(cudaMalloc((double **) &d_deriv_rdm, sizeof(double) * number_pts_iter));
    // Allocate host to transfer it to `h_kinetic_density`
    std::vector<double> h_deriv_rdm(number_pts_iter);
    // For each derivative, calculate the derivative of electron density seperately.
    for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
      // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
      double *d_ith_deriv = &d_deriv_contractions[i_deriv * number_pts_iter * knbasisfuncs];

      // Matrix multiple one-rdm with the ith derivative of contractions
      double alpha = 1.0;
      double beta = 0.0;
      chemtools::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              number_pts_iter, knbasisfuncs, knbasisfuncs,
                                              &alpha, d_ith_deriv, number_pts_iter,
                                              d_one_rdm, knbasisfuncs, &beta,
                                              d_temp_rdm_derivs, number_pts_iter));

      // Do a hadamard product with the original contractions.
      dim3 threadsPerBlock2(320);
      dim3 grid2((number_pts_iter * knbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
      chemtools::hadamard_product<<<grid2, threadsPerBlock2>>>(
          d_temp_rdm_derivs, d_ith_deriv, knbasisfuncs, number_pts_iter
      );

      // Take the sum.
      thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
      double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
      chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N,
                                              number_pts_iter, knbasisfuncs,
                                              &alpha, d_temp_rdm_derivs, number_pts_iter, deviceVecPtr, 1, &beta,
                                              d_deriv_rdm, 1));

      // Multiply by 0.5
      dim3 threadsPerBlock3(320);
      dim3 grid3((number_pts_iter + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
      chemtools::multiply_scalar<<< grid3, threadsPerBlock3>>>(d_deriv_rdm, 0.5, number_pts_iter);

      chemtools::cuda_check_errors(cudaMemcpy(h_deriv_rdm.data(), d_deriv_rdm,
                                           sizeof(double) * number_pts_iter, cudaMemcpyDeviceToHost));

      // Add to h_laplacian
      for(size_t i = index_to_copy; i < index_to_copy + number_pts_iter; i++) {
        h_kinetic_dens[i] += h_deriv_rdm[i - index_to_copy];
      }

      // Free up memory in this iteration for the next calculation of the derivative.
      all_ones.clear();
      all_ones.shrink_to_fit();
    }
    cudaFree(d_temp_rdm_derivs);
    cudaFree(d_deriv_contractions);
    cudaFree(d_deriv_rdm);

    // Update lower-bound of the grid for the next iteration
    index_to_copy += number_pts_iter;
    i_iter += 1;  // Update the index for each iteration.
  }

  cudaFree(d_one_rdm);
  return h_kinetic_dens;
}


__host__ std::vector<double> chemtools::evaluate_positive_definite_kinetic_density(
    chemtools::IOData &iodata, const double *h_points, const int knumb_points
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> kinetic_density = chemtools::evaluate_pos_def_kinetic_density_on_any_grid_handle(
      handle, iodata, h_points, knumb_points
  );
  cublasDestroy(handle); // cublas handle is no longer needed infact most of
  return kinetic_density;
}


__host__ std::vector<double> chemtools::evaluate_general_kinetic_energy_density(
    chemtools::IOData &iodata, const double alpha, const double *h_points, const int knumb_points
){
  // Evaluate \nabla^2 \rho
  std::vector<double> h_laplacian = chemtools::evaluate_laplacian(
      iodata, h_points, knumb_points
      );

  //  Evaluate t_{+}
  std::vector<double> h_pos_def_kinetic_energy = chemtools::evaluate_positive_definite_kinetic_density(
      iodata, h_points, knumb_points
      );

  // Add them t_{+} + \alpha \nabla^2 \rho
  for(int i = 0; i < knumb_points; i++){
    h_pos_def_kinetic_energy[i] += h_laplacian[i] * alpha;
  }

  return h_pos_def_kinetic_energy;
}
