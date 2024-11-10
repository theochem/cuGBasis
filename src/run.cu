
#include "eval_density.cuh"
#include "basis_to_gpu.cuh"
#include "cuda_utils.cuh"
#include "run.cuh"

using namespace  chemtools;

__host__ void chemtools::evaluate_scalar_quantity_density(
    const MolecularBasis& basis,
          bool            do_segmented_basis,
    const bool            disp,
          double*         d_output_iter,
    const double*         d_points_iter,
    const int             knumb_points_iter,
    const int             k_total_numb_contractions,
          dim3            threadsPerBlock,
          dim3            grid,
      cudaFuncCache l1_over_shared
) {
  // Set the temprory kernel to prefer L1 cache over shared memory.
  chemtools::cuda_check_errors(cudaFuncSetCacheConfig(chemtools::eval_AOs_from_constant_memory_on_any_grid,
                                                      l1_over_shared));

  // Start at the first shell
  std::size_t i_shell = 0;                              // Controls which shells are in constant memory
  std::size_t numb_basis_funcs_completed = 0;           // Controls where to update the next set of contractions
  std::array<std::size_t, 2> i_shell_and_numb_basis{};  // Used to update how many shells and contractions are in
  //    constant memory.
  // Increment through each shell that can fit inside constant memory.
  while (i_shell < basis.numb_contracted_shells()) {
    // Transfer the basis-set to constant memory
    i_shell_and_numb_basis = chemtools::add_mol_basis_to_constant_memory_array(
        basis, do_segmented_basis, disp, i_shell
    );

    // Evaluate the function over the GPU
    chemtools::eval_AOs_from_constant_memory_on_any_grid<<<grid, threadsPerBlock>>>(
        d_output_iter, d_points_iter, knumb_points_iter, k_total_numb_contractions,
            static_cast<int>(numb_basis_funcs_completed));
    cudaDeviceSynchronize();

    // Update to the next begining of the shells and contractions
    i_shell = i_shell_and_numb_basis[0];
    numb_basis_funcs_completed += i_shell_and_numb_basis[1];
  }
}