
#include "eval_mo.cuh"
#include "cuda_utils.cuh"
#include "basis_to_gpu.cuh"
#include "eval.cuh"

using namespace chemtools;

__host__ std::vector<double> chemtools::eval_MOs(
    IOData& iodata, const double* h_points, int n_pts, const std::string& spin
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return ((mem / sizeof(double))  - numb_basis * numb_basis) / (2 * numb_basis);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Resulting Molecular orbitals (M, N) col-order
    std::vector<double> h_MOs_col(n_pts * n_basis);
    
    // Iterate through each chunk of the data set.
    size_t index_to_copy = 0;  
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        // Calculate number of points to do
        size_t npts_iter    = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );
        size_t AO_data_size = npts_iter * n_basis * sizeof(double);
        
        // Allocate device memory for contractions row-major (M, N)
        double *d_AOs = nullptr;
        CUDA_CHECK(cudaMalloc((double **) &d_AOs, AO_data_size));
        CUDA_CHECK(cudaMemset(d_AOs, 0, AO_data_size));
        
        // Allocate points and copy grid points column-order
        double *d_pts;
        CUDA_CHECK(cudaMalloc((double **) &d_pts, sizeof(double) * 3 * npts_iter));
        #pragma unroll
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpy(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Evaluate AOs.
        constexpr int THREADS_PER_BLOCK = 128;
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks((npts_iter + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho",
            d_AOs,
            d_pts,
            npts_iter,
            n_basis,
            threads,
            blocks
        );
        cudaFree(d_pts);
        
        // Allocate device memory for the one_rdm. Number AO = Number MO
        double *d_MO_coeffs;
        CUDA_CHECK(cudaMalloc((double **) &d_MO_coeffs, n_basis * n_basis * sizeof(double)));
        CUBLAS_CHECK(cublasSetMatrix(
            iodata.GetOneRdmShape(),
            iodata.GetOneRdmShape(),
            sizeof(double),
            iodata.GetMOCoeffs(spin),
            iodata.GetOneRdmShape(),
            d_MO_coeffs,
            iodata.GetOneRdmShape()
        ));

        // Molecular Orbitals (M, N) row
        double *d_MOs;
        CUDA_CHECK(cudaMalloc((double **) &d_MOs, AO_data_size));

        // Matrix mult. of one rdm with the contractions array. Everything is in row major order.
        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            npts_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
            &alpha,
            d_AOs, npts_iter,
            d_MO_coeffs, iodata.GetOneRdmShape(),
            &beta,
            d_MOs, npts_iter
        ));
        cudaFree(d_MO_coeffs);
        
        // Transfer electron density from device memory to host memory.
        //    Since I'm computing a sub-grid at a time, need to update the index h_electron_density, accordingly.
        //    Note that d_mol_orbs is in row-major order with shape (M, N)
        std::vector<double> h_mol_orbitals_row(n_basis * npts_iter);
        cuda_check_errors(cudaMemcpy(h_mol_orbitals_row.data(),
                                     d_MOs,
                                     sizeof(double) * n_basis * npts_iter, cudaMemcpyDeviceToHost));
        cudaFree(d_MOs);
        
        // Transfer from row to column major order;
        for(int i_row = 0; i_row < n_basis; i_row++){
            for(int j_col = 0; j_col < npts_iter; j_col++) {
                h_MOs_col[n_basis * index_to_copy + (j_col * n_basis + i_row)] = h_mol_orbitals_row[i_row * npts_iter + j_col];
            }
        }
        
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter += 1;  // Update the index for each iteration.
    }
    cublasDestroy(handle);
    
    return h_MOs_col;
}


__host__ std::vector<double> chemtools::eval_MOs_derivs(
          IOData &iodata,
    const double *h_points,
    const int    n_pts)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return ((mem / sizeof(double))  - numb_basis * numb_basis) / (4 * numb_basis);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Resulting Molecular orbitals (3, N, M) col-order
    std::vector<double> h_MOs_col(3 * n_pts * n_basis);
    
    // Iterate through each chunk of the data set.
    size_t index_to_copy = 0;
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        size_t npts_iter = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );
        size_t AO_deriv_size = sizeof(double) * 3 * npts_iter * n_basis;
        
        // Allocate device memory for contractions row-major (M, N)
        double *d_AOs_derivs = nullptr;
        CUDA_CHECK(cudaMalloc((double **) &d_AOs_derivs, AO_deriv_size));
        CUDA_CHECK(cudaMemset(d_AOs_derivs, 0, AO_deriv_size));
        
        // Allocate points and copy grid points column-order
        double *d_pts;
        CUDA_CHECK(cudaMalloc((double **) &d_pts, sizeof(double) * 3 * npts_iter));
        #pragma unroll
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpy(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Evaluate Derivatives of AOs.
        constexpr int THREADS_PER_BLOCK = 128;
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks((npts_iter + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho_deriv",
            d_AOs_derivs,
            d_pts,
            npts_iter,
            n_basis,
            threads,
            blocks
        );
        cudaFree(d_pts);
        
        // Allocate device memory for the one_rdm. Number AO = Number MO
        double *d_one_rdm;
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
        
        // Resulting ith MO Deriv
        double *d_MOs_derivi;
        CUDA_CHECK(cudaMalloc((double **) &d_MOs_derivi, sizeof(double) * npts_iter * n_basis));
        
        // For each derivative, calculate the derivative of MOs.
        double alpha = 1.0, beta = 0.0;
        #pragma unroll 3
        for (int i_deriv = 0; i_deriv < 3; i_deriv++) {
            // Ith deriv of AOs (M, N) row-order
            double *d_ith_deriv = &d_AOs_derivs[i_deriv * npts_iter * n_basis];
            
            // Matrix multiple one-rdm with the ith derivative of contractions
            CUBLAS_CHECK(cublasDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                npts_iter, n_basis, n_basis,
                &alpha,
                d_ith_deriv, npts_iter,
                d_one_rdm, n_basis,
                &beta,
                d_MOs_derivi, npts_iter
            ));
            
            // Transpose and over-write d_AOs_derivs so its (M, N) row-major
            array_transpose(
                handle, &d_AOs_derivs[i_deriv * npts_iter * n_basis],
                d_MOs_derivi, n_basis, npts_iter
                );
            
            // Copy it over
            cuda_check_errors(cudaMemcpy(&h_MOs_col[i_deriv * n_basis * n_pts +   n_basis * index_to_copy],
                                         d_ith_deriv,
                                         sizeof(double) * n_basis * npts_iter, cudaMemcpyDeviceToHost));
        }
        cudaFree(d_MOs_derivi);
        cudaFree(d_one_rdm);
        cudaFree(d_AOs_derivs);
        
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter += 1;  // Update the index for each iteration.
    }
    cublasDestroy(handle);
    
    return h_MOs_col;
}


__host__ std::vector<double> chemtools::eval_MOs_second_derivs(
    IOData& iodata, const double* h_points, const int n_pts
)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return ((mem / sizeof(double))  - numb_basis * numb_basis) / (4 * numb_basis);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Resulting Molecular orbitals (6, N, M) col-order
    std::vector<double> h_MOs_sec_col(6 * n_pts * n_basis);
    
    // Iterate through each chunk of the data set.
    size_t index_to_copy = 0;
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        size_t npts_iter = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );
        size_t AO_deriv_size = sizeof(double) * 6 * npts_iter * n_basis;
        
        // Allocate device memory for contractions row-major (M, N)
        double *d_AOs_sec_derivs = nullptr;
        CUDA_CHECK(cudaMalloc((double **) &d_AOs_sec_derivs, AO_deriv_size));
        CUDA_CHECK(cudaMemset(d_AOs_sec_derivs, 0, AO_deriv_size));
        
        // Allocate points and copy grid points column-order
        double *d_pts;
        CUDA_CHECK(cudaMalloc((double **) &d_pts, sizeof(double) * 3 * npts_iter));
        #pragma unroll
        for(int coord = 0; coord < 3; coord++) {
            CUDA_CHECK(cudaMemcpy(
                &d_pts[coord * npts_iter],
                &h_points[coord * n_pts + index_to_copy],
                sizeof(double) * npts_iter,
                cudaMemcpyHostToDevice)
            );
        }
        
        // Evaluate Derivatives of AOs.
        constexpr int THREADS_PER_BLOCK = 128;
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks((npts_iter + THREADS_PER_BLOCK - 1) / (THREADS_PER_BLOCK));
        evaluate_scalar_quantity_density(
            molbasis,
            false,
            false,
            "rho_hess",
            d_AOs_sec_derivs,
            d_pts,
            npts_iter,
            n_basis,
            threads,
            blocks
        );
        cudaFree(d_pts);
        
        // Allocate device memory for the one_rdm. Number AO = Number MO
        double *d_one_rdm;
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
        
        // Resulting ith MO Deriv
        double *d_MOs_sec_derivi;
        CUDA_CHECK(cudaMalloc((double **) &d_MOs_sec_derivi, sizeof(double) * npts_iter * n_basis));
        
        // From atomic orbitals calculate MO second- derivatives
        double alpha = 1.0;
        double beta = 0.0;
        int i_sec_deriv = 0;
        for(int i_deriv = 0; i_deriv < 3; i_deriv++) {
            for (int j_deriv = i_deriv; j_deriv < 3; j_deriv++) {
                // Get the second derivative: `i_deriv`th and `j_deriv`th derivative.
                double *d_sec_deriv = &d_AOs_sec_derivs[i_sec_deriv * npts_iter * n_basis];
                
                // Matrix multiple one-rdm with the ith derivative of contractions
                CUBLAS_CHECK(cublasDgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    npts_iter, n_basis, n_basis,
                    &alpha,
                    d_sec_deriv, npts_iter,
                    d_one_rdm, n_basis,
                    &beta,
                    d_MOs_sec_derivi, npts_iter
                ));
                
                // Transpose and over-write d_AOs_derivs so its (M, N) row-major
                array_transpose(
                    handle, &d_AOs_sec_derivs[i_sec_deriv * npts_iter * n_basis],
                    d_MOs_sec_derivi, n_basis, npts_iter
                );
                
                // Copy it over
                cuda_check_errors(cudaMemcpy(&h_MOs_sec_col[i_sec_deriv * n_basis * n_pts +  n_basis * index_to_copy],
                                             d_sec_deriv,
                                             sizeof(double) * n_basis * npts_iter, cudaMemcpyDeviceToHost));
                i_sec_deriv++;
            }
        }
        cudaFree(d_MOs_sec_derivi);
        cudaFree(d_one_rdm);
        cudaFree(d_AOs_sec_derivs);
    
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter += 1;  // Update the index for each iteration.
    }
    cublasDestroy(handle);
    
    return h_MOs_sec_col;
}