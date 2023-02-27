
#include <thrust/device_vector.h>
#include "cublas_v2.h"

#include "../include/evaluate_kinetic_dens.cuh"
#include "../include/evaluate_gradient.cuh"
#include "../include/evaluate_laplacian.cuh"
#include "../include/cuda_utils.cuh"


__host__ std::vector<double> gbasis::evaluate_pos_def_kinetic_density_on_any_grid_handle(
    cublasHandle_t &handle, gbasis::IOData &iodata, const double *h_points, const int knumb_points
) {
  // Set cache perference to L1
  cudaFuncSetCacheConfig(
      gbasis::evaluate_derivatives_contractions_from_constant_memory, cudaFuncCachePreferL1
  );

  // Get the molecular basis from iodata and put it in constant memory of the gpu.
  gbasis::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
  //gbasis::add_mol_basis_to_constant_memory_array(molecular_basis, false, false);
  int knbasisfuncs = molecular_basis.numb_basis_functions();

  // Electron density in global memory and create the handles for using cublas.
  std::vector<double> h_kinetic_dens(knumb_points);

  // Transfer grid points to GPU, this is in column order with shape (N, 3)
  double* d_points;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_points));
  gbasis::cuda_check_errors(cudaMemcpy(d_points, h_points,sizeof(double) * 3 * knumb_points, cudaMemcpyHostToDevice));

  // Evaluate derivatives of each contraction this is in row-order (3, M, N), where M =number of basis-functions.
  double* d_deriv_contractions;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_deriv_contractions, sizeof(double) * 3 * knumb_points * knbasisfuncs));
  dim3 threadsPerBlock(128);
  dim3 grid((knumb_points + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::evaluate_derivatives_contractions_from_constant_memory<<<grid, threadsPerBlock>>>(
      d_deriv_contractions, d_points, knumb_points, knbasisfuncs
  );

  // Free up points memory in device/gpu memory.
  cudaFree(d_points);

  // Transfer one-rdm from host/cpu memory to device/gpu memory.
  double *d_one_rdm;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_one_rdm, knbasisfuncs * knbasisfuncs * sizeof(double)));
  gbasis::cublas_check_errors(cublasSetMatrix(iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
                                              sizeof(double), iodata.GetMOOneRDM(),
                                              iodata.GetOneRdmShape(), d_one_rdm,iodata.GetOneRdmShape()));

  // Allocate memory to hold the matrix-multiplcation between d_one_rdm and each `i`th derivative (i_deriv, M, N)
  double *d_temp_rdm_derivs;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_temp_rdm_derivs, sizeof(double) * knumb_points * knbasisfuncs));
  // Allocate device memory for the derivative of the one-electorn rdm column-major order.
  double *d_deriv_rdm;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_deriv_rdm, sizeof(double) * knumb_points));
  // Allocate host to transfer it to `h_kinetic_density`
  std::vector<double> h_deriv_rdm(knumb_points);
  // For each derivative, calculate the derivative of electron density seperately.
  for(int i_deriv = 0; i_deriv < 3; i_deriv++) {
    // Get the ith derivative of the contractions with shape (M, N) in row-major order, N=numb pts, M=numb basis funcs
    double* d_ith_deriv = &d_deriv_contractions[i_deriv * knumb_points * knbasisfuncs];

    // Matrix multiple one-rdm with the ith derivative of contractions
    double alpha = 1.0;
    double beta = 0.0;
    gbasis::cublas_check_errors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            knumb_points, knbasisfuncs, knbasisfuncs,
                                            &alpha, d_ith_deriv, knumb_points,
                                            d_one_rdm, knbasisfuncs, &beta,
                                            d_temp_rdm_derivs, knumb_points));

    // Do a hadamard product with the original contractions.
    dim3 threadsPerBlock2(320);
    dim3 grid2((knumb_points * knbasisfuncs + threadsPerBlock.x - 1) / (threadsPerBlock.x));
    gbasis::hadamard_product<<<grid2, threadsPerBlock2>>>(
        d_temp_rdm_derivs, d_ith_deriv, knbasisfuncs, knumb_points
    );

    // Take the sum.
    thrust::device_vector<double> all_ones(sizeof(double) * knbasisfuncs, 1.0);
    double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
    gbasis::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N,
                                            knumb_points, knbasisfuncs,
                                            &alpha, d_temp_rdm_derivs, knumb_points, deviceVecPtr, 1, &beta,
                                            d_deriv_rdm, 1));

    // Multiply by 0.5
    dim3 threadsPerBlock3(320);
    dim3 grid3((knumb_points + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
    gbasis::multiply_scalar<<< grid3, threadsPerBlock3>>>(d_deriv_rdm, 0.5, knumb_points);


    gbasis::cuda_check_errors(cudaMemcpy(h_deriv_rdm.data(), d_deriv_rdm,
                                         sizeof(double) * knumb_points, cudaMemcpyDeviceToHost));

    // Add to h_laplacian
    for(int i = 0; i < knumb_points; i++) {
      h_kinetic_dens[i] += h_deriv_rdm[i];
    }

    // Free up memory in this iteration for the next calculation of the derivative.
    all_ones.clear();
    all_ones.shrink_to_fit();
  }

  cudaFree(d_temp_rdm_derivs);
  cudaFree(d_one_rdm);
  cudaFree(d_deriv_contractions);
  cudaFree(d_deriv_rdm);

  return h_kinetic_dens;
}


__host__ std::vector<double> gbasis::evaluate_positive_definite_kinetic_density(
    gbasis::IOData &iodata, const double *h_points, const int knumb_points
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> kinetic_density = gbasis::evaluate_pos_def_kinetic_density_on_any_grid_handle(
      handle, iodata, h_points, knumb_points
  );
  cublasDestroy(handle); // cublas handle is no longer needed infact most of
  return kinetic_density;
}


__host__ std::vector<double> gbasis::evaluate_general_kinetic_energy_density(
    gbasis::IOData &iodata, const double alpha, const double *h_points, const int knumb_points
){
  // Evaluate \nabla^2 \rho
  std::vector<double> h_laplacian = gbasis::evaluate_laplacian(
      iodata, h_points, knumb_points
      );

  //  Evaluate t_{+}
  std::vector<double> h_pos_def_kinetic_energy = gbasis::evaluate_positive_definite_kinetic_density(
      iodata, h_points, knumb_points
      );

  // Add them t_{+} + \alpha \nabla^2 \rho
  for(int i = 0; i < knumb_points; i++){
    h_pos_def_kinetic_energy[i] += h_laplacian[i] * alpha;
  }

  return h_pos_def_kinetic_energy;
}
