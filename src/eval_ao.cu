

#include "eval_ao.cuh"
#include "cuda_utils.cuh"
#include "basis_to_gpu.cuh"
#include "cuda_basis_utils.cuh"
#include "eval.cuh"

using namespace chemtools;

__host__ std::vector<double> eval_AOs(
    IOData& iodata, const double* h_points, const int n_pts
){
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const MolecularBasis molbasis = iodata.GetOrbitalBasis();
    const int            n_basis  = molbasis.numb_basis_functions();
    
    // Calculate Optimal Memory Chunks
    const size_t MAX_PTS_PER_ITER = 64 * 64 * 32  ;
    auto   chunks     = GpuMemoryPartitioner::compute(
        n_basis,
        [](size_t mem, size_t numb_basis){
          return ((mem / sizeof(double))) / (2 * numb_basis);
        },
        n_pts,
        MAX_PTS_PER_ITER
    );
    
    // Resulting Atomic orbitals (M, N) col-order
    std::vector<double> h_AOs_col(n_pts * n_basis);
    
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
        
        double *d_AO_col;
        CUDA_CHECK(cudaMalloc((double **) &d_AO_col, sizeof(double) * n_basis * npts_iter));
        
        // Transfer electron density from device memory to host memory.
        array_transpose(
            handle, d_AO_col,
            d_AOs, n_basis, npts_iter
        );
        
        // Copy it over
        CUDA_CHECK(cudaMemcpy(&h_AOs_col[n_basis * index_to_copy],
                              d_AOs,
                              sizeof(double) * n_basis * npts_iter, cudaMemcpyDeviceToHost));
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter += 1;  // Update the index for each iteration.
    }
    cublasDestroy(handle);
    
    return h_AOs_col;
}
