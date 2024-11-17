
#include <thrust/device_vector.h>
#include "cublas_v2.h"

#include "eval_kin_energ.cuh"
#include "eval_lap.cuh"
#include "cuda_utils.cuh"
#include "basis_to_gpu.cuh"
#include "eval.cuh"

using namespace chemtools;

__host__ std::vector<double> chemtools::evaluate_pos_def_kinetic_density_on_any_grid_handle(
          cublasHandle_t &handle,
          IOData         &iodata,
    const double         *h_points,
          int            n_pts
) {
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();

    // Electron density in global memory and create the handles for using cublas.
    std::vector<double> h_kinetic_dens(n_pts);
    
    // Calculate Optimal Memory Chunks
    //  Solve for N (12MN + 9N + N + M) * 8 bytes = Free memory (in bytes)
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return (
              (mem  / sizeof(double))  - numb_basis - numb_basis * numb_basis
          ) /  (3 * numb_basis + numb_basis);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Allocate some memory outside the loop to avoid the slow cudaMalloc cudaFree
    double *d_one_rdm = nullptr;
    CUDA_CHECK(cudaMalloc((double **) &d_one_rdm, n_basis * n_basis * sizeof(double)));
    CUBLAS_CHECK(cublasSetMatrix(
        iodata.GetOneRdmShape(),
        iodata.GetOneRdmShape(),
        sizeof(double),
        iodata.GetMOOneRDM(),
        iodata.GetOneRdmShape(),
        d_one_rdm,
        iodata.GetOneRdmShape()
    ));
    
    // Iterate through each chunk
    size_t index_to_copy = 0;
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        size_t npts_iter = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );
        
        // Allocate points and copy grid points column-order
        double *d_pts;
        CUDA_CHECK(cudaMalloc((double **) &d_pts, sizeof(double) * 3 * npts_iter));
        #pragma unroll 3
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpy(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Evaluate AOs derivatives  row-order (3, M, N)
        double *d_AOs_deriv;
        CUDA_CHECK(cudaMalloc((double **) &d_AOs_deriv, sizeof(double) * 3 * npts_iter * n_basis));
        dim3 threadsPerBlock(128);
        dim3 grid((npts_iter + threadsPerBlock.x - 1) / (threadsPerBlock.x));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho_deriv",
            d_AOs_deriv,
            d_pts,
            npts_iter,
            n_basis,
            threadsPerBlock,
            grid
        );
        // Free up points memory in device/gpu memory.
        cudaFree(d_pts);
        
        // ith derivative of MOs (i_deriv, M, N)
        double *d_MOs_deriv;
        double *d_deriv_rdm;
        CUDA_CHECK(cudaMalloc((double **) &d_MOs_deriv, sizeof(double) * npts_iter * n_basis));
        CUDA_CHECK(cudaMalloc((double **) &d_deriv_rdm, sizeof(double) * npts_iter));
        
        // Allocate host to transfer it to `h_kinetic_density`
        std::vector<double> h_deriv_rdm(npts_iter);
        
        // For each derivative, calculate the derivative of electron density seperately.
        for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
            // ith derivative AOs (M, N) row-major order
            double *d_ith_deriv = &d_AOs_deriv[i_deriv * npts_iter * n_basis];
            
            // Matrix multiple one-rdm with the ith derivative of contractions
            double alpha = 1.0, beta = 0.0;
            CUBLAS_CHECK(cublasDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                npts_iter, n_basis, n_basis,
                &alpha,
                d_ith_deriv, npts_iter, d_one_rdm, n_basis,
                &beta,
                d_MOs_deriv, npts_iter)
            );
            
            // Do a hadamard product with the original contractions.
            dim3 threadsPerBlock2(320);
            dim3 grid2((npts_iter * n_basis + threadsPerBlock.x - 1) / (threadsPerBlock.x));
            hadamard_product<<<grid2, threadsPerBlock2>>>(
                d_MOs_deriv, d_ith_deriv, n_basis, npts_iter
            );
            
            // Take the sum.
            thrust::device_vector<double> all_ones(sizeof(double) * n_basis, 1.0);
            double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
            CUBLAS_CHECK(cublasDgemv(
                handle, CUBLAS_OP_N,
                npts_iter, n_basis,
                &alpha, d_MOs_deriv,
                npts_iter, deviceVecPtr, 1,
                &beta, d_deriv_rdm, 1)
            );
            
            // Multiply by 0.5
            dim3 threadsPerBlock3(320);
            dim3 grid3((npts_iter + threadsPerBlock3.x - 1) / (threadsPerBlock3.x));
            multiply_scalar<<< grid3, threadsPerBlock3>>>(d_deriv_rdm, 0.5, npts_iter);
            
            // Copy
            CUDA_CHECK(cudaMemcpy(
                h_deriv_rdm.data(),
                d_deriv_rdm,
                sizeof(double) * npts_iter,
                cudaMemcpyDeviceToHost)
            );
            
            // Add to h_laplacian
            for(size_t i = index_to_copy; i < index_to_copy + npts_iter; i++)
                h_kinetic_dens[i] += h_deriv_rdm[i - index_to_copy];
            
            // Free up memory in this iteration for the next calculation of the derivative.
            all_ones.clear();
            all_ones.shrink_to_fit();
        }
        cudaFree(d_MOs_deriv);
        cudaFree(d_AOs_deriv);
        cudaFree(d_deriv_rdm);
        
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter++;
    }
    
    cudaFree(d_one_rdm);
    return h_kinetic_dens;
}


__host__ std::vector<double> chemtools::evaluate_positive_definite_kinetic_density(
          IOData &iodata,
    const double *h_points,
          int    n_pts
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<double> kinetic_density = evaluate_pos_def_kinetic_density_on_any_grid_handle(
      handle, iodata, h_points, n_pts
  );
  cublasDestroy(handle);
  return kinetic_density;
}


__host__ std::vector<double> chemtools::evaluate_general_kinetic_energy_density(
          IOData &iodata,
          double alpha,
    const double *h_points,
          int    n_pts
){
  // Evaluate \nabla^2 \rho
  std::vector<double> h_laplacian = evaluate_laplacian(
      iodata, h_points, n_pts
      );

  //  Evaluate t_{+}
  std::vector<double> h_pos_def_kinetic_energy = evaluate_positive_definite_kinetic_density(
      iodata, h_points, n_pts
      );

  // Add them t_{+} + \alpha \nabla^2 \rho
  for(int i = 0; i < n_pts; i++){
    h_pos_def_kinetic_energy[i] += h_laplacian[i] * alpha;
  }

  return h_pos_def_kinetic_energy;
}
