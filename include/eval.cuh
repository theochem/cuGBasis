#ifndef CUGBASIS_INCLUDE_EVAL_CUH_
#define CUGBASIS_INCLUDE_EVAL_CUH_

#include "contracted_shell.h"
#include "cuda_utils.cuh"

namespace chemtools {
struct GpuMemoryPartitioner {
    size_t free_mem;
    size_t total_mem;
    size_t pts_per_iter;
    
    /// Given number of free-memory and number of basis-functions calculates number of
    ///     Points to do each iteration.
    using CalculatorFunction = std::function<size_t(size_t, size_t)>;
    
    static GpuMemoryPartitioner compute(
        size_t             nbasis,
        CalculatorFunction func_pts_per_iter,
        size_t             total_pts,
        size_t             max_pts_per_iter = 0)
    {
        const size_t MEMORY_BUFFER = 500'000'000;  // 0.5 GB
        
        // Get available GPU memory (in bytes);
        size_t free_mem  = 0;
        size_t total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        // Calculate how many points we can compute with 0.5 GB safety margin
        size_t usable_memory = free_mem - MEMORY_BUFFER;
        size_t pts_per_iter = func_pts_per_iter(usable_memory, nbasis);
        if (pts_per_iter == 0) {
            throw std::runtime_error("Number of points of each chunk cannot be zero");
        }
        
        // Cap the number of points, if the argument is provided and pts_per_iter is less than total number of pts.
        //     Used to improve efficiency, as sometimes the optimal number of points to calculate
        //     per iteration does not lead to the fastest code (particularly matrix-multiplcation).
        //     If pts_per_iter is greater than total number of points, then just do the total number of points in one
        //          iteration, for small basis-sets this should be ideal.
        if (max_pts_per_iter != 0 && pts_per_iter < total_pts)
            pts_per_iter = std::min(max_pts_per_iter, pts_per_iter);
        else
            pts_per_iter = total_pts;
        
        return GpuMemoryPartitioner{free_mem, total_mem, pts_per_iter};
    }
};


  __host__ void evaluate_scalar_quantity_density(
      const MolecularBasis& basis,
      bool            do_segmented_basis,
      const bool            disp,
      const std::string&     type,
      double*         d_output_iter,
      const double*         d_points_iter,
      const int             knumb_points_iter,
      const int             k_total_numb_contractions,
      dim3            threadsPerBlock,
      dim3            grid,
      cudaFuncCache l1_over_shared = cudaFuncCachePreferL1
  );
}
#endif //CUGBASIS_INCLUDE_EVAL_CUH_
