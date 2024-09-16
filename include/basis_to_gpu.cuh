#ifndef CHEMTOOLS_CUDA_INCLUDE_BASIS_TO_GPU_H_
#define CHEMTOOLS_CUDA_INCLUDE_BASIS_TO_GPU_H_

#include <functional>
#include <vector>
#include <string>
#include <unordered_map>

#include "contracted_shell.h"

// Stores constant memory for the NVIDIA GPU.
extern __constant__ double g_constant_basis[7500];

// Create a function pointer type definition
typedef void (*d_func_t)(double*, const double*, const int, const int, const int);

namespace chemtools {
/***
 *
 *
 * Wave-function
 *
 */

/**
 * Puts molecular basis into constant memory of the NVIDIA GPU as a straight array. Useful when one
 *  just wants to iterate through the array in a row fashion, e.g. evaluating electron density/contraction array.
 *  The downside is the inability to jump to any contracted shell.
 *
 * @param[in] basis The molecular basis set information as a collection of contracted shells.
 * @param[out] array of length two.  The first element is the index of the shell that was included in the constant
 *             memory. If this index is equal to (number of contractions minus one) then it implies the entire
 *             molecular basis is in constant memory.  The second element is the number of contractions that are
 *             going to be evaluted if one iterates through constant memory.
 * @note Every basis-set information is casted to a double.
 * @note Storing it as a decontracted basis, i.e. every contraction shell is segmented, i.e. has only one angular
 *       momentum associated with it, is that it reduced the amount of for-loops inside the cuda function. This
 *       is particularly useful when doing integration like electrostatic potential.
 * @note The following explains how memory is placed is with constant memory.
 *          M := numb_segmented_shell inside contracted shell.
 *          K := Number of primitives per Mth segmented shell.
 *
 *          | numb_contracted_shells | 4 byte blank | atom_coord_x | atom_coord_y | atom_coord_z | M | 4B Blank | K | ...
 *          4B Blank | exponent_1 | ... | exponent_K | angmom_1 | coeff_11 | ... | coeff_K1 | ... | angmom_M | coeff_1M |..
 *          coeff_KM | K2 | 4B Blank | exponent_2 | same pattern repeat...
 *
 *          Segmented Basis mode:
 *  | numb_contracted_shells | 4 byte blank | atom_coord_x | atom_coord_y | atom_coord_z | K | ...
 *     4B Blank | angmom_1 | exponent_1 | ... | exponent_K |  coeff_1 | ... | coeff_K | atom_coord_x | atom_coord_y |
 *     atom_coord_z |  K2 | 4B Blank |  angmom_2 | exponent_1 | ... | exponent_K2 | coeff_1 | same pattern repeat...
 */
__host__ std::array<std::size_t, 2> add_mol_basis_to_constant_memory_array(
    const chemtools::MolecularBasis& basis,
    bool do_segmented_basis = false,
    const bool disp = false,
    const std::size_t i_shell_start = 0
);

/**
 * Puts molecular basis into constant memory of the NVIDIA GPU as a straight array with consideration
 *  to the fact that one needs to be able to jump to any contracted shell. This puts less strain
 *  under the registers
 *  .
 * @param[in] basis The molecular basis set information as a collection of contracted shells.
 * @note The following explains how memory is placed within constant memory.
 *       N := number of contracted shells.
 *       M_i := numb_segmented_shell inside ith contracted shell where 1 <= i <= N.
 *       K_i := Number of primitives per ith segmented shell where 1 <= i <= N.
*         Number of primitives is consistent inside a contracted hell.
 *  | N | 4 byte blank | M_1 | 4 byte blank | K_1 | 4 byte blank | .... | M_N | 4B | K_N | 4B
 *      | *atom_coord_x | atom_coord_y | atom_coord_z |
 *
 *  The first series of integers from N to K_N, is all the information you need to jump to any contracted shell.
 *      To reach *atom_coord_x, it is precisely the index 1 + 2 * N.
 *
 */
__host__ void add_mol_basis_to_constant_memory_array_access(chemtools::MolecularBasis basis);


__host__ void evaluate_scalar_quantity(
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
    cudaFuncCache l1_over_shared = cudaFuncCachePreferL1
);



/***
 *
 * Promolecular
 *
 */

/**
 * Puts promolecular coefficents/exponents into constant memory of the NVIDIA GPU as a straight array.
 *   Only has s-type and p-type Gaussians
 *   Useful when one just wants to iterate through the array in a row fashion
 *
 *  So far the elemnts provided are
 *        H  C  N  O  F  S  P  Cl
 *
 * @param[in] basis The molecular basis set information as a collection of contracted shells.
 * @param[in] atom_coords Atomic coordinates of size (M, 3) in row-major order
 * @param[in] atom_numbers Atomic atom_numbers of size(M,) used to identify what element it is
 * @param[in] natoms The number of atoms M.
 * @param[in] promol_coeffs The promolecular coefficients of s-type and p-type Gaussians.
 * @param[in] promol_exps The promolecular exponents of s-type and p-type Gaussians.
 * @param[in] i_atom_start The atomic coordinates to put in constant memory, used to terminate the process
 * @param[in] index_atom_coords The Index within constant memory where atomic coordinates should start.
 *                              If zero, then it means it is the first time.
 *
 * @param[out] array of two ints,
 *             First int, tells the index where the promolecular basis ends, i.e. index within constant memory
 *                where the atomic coordinates should start, needed for GPU code
 *             Second int, tells which atom to do next in the next cycle, needed for Host Code, if this
 *                is equal to the number of atoms, then it implies all atoms were completed.
 * @note Every basis-set information is casted to a double.
 * @note The following explains how memory is placed is with constant memory.
 *          i^E := Index of element `E` within constant memory where the promolecular is listed.
 *          M^E_S := Number of s-type coefficeints for elemenmt E
 *          M^E_P := Number of p-type coefficeints for element E
 *
 *          | i^H | 4 byte blank | i^C | 4B blank | i^N | 4B blank | ... | i^Cl | 4B Blank
 *           M^H_S | 4B  | coeff_1 | exp_1 | coeff_2 | exp_2 | ... | M^H_P | 4B byte blank | coeff_1 | exp_1 | ....
 *           Pattern repeats for the next element, starting with M^E_S
 *
 *          After that the Number of atoms (denoted as M) within constant memory is placed, then
 *          the Index of i^E is placed where E is determined from the atomic charge,
 *          then atomic coordinates for that element E is placed.
 *          | M | 1 (for i^C) | C_x | C_y | C_z | 0 (for i^H) | H_x | H_y | H_z | ....
 *
 */
__host__ std::array<std::size_t, 2> add_promol_basis_to_constant_memory_array(
    const double* const atom_coords,
    const long int *const atom_numbers,
    int natoms,
    const std::unordered_map<std::string, std::vector<double>>& promol_coeffs,
    const std::unordered_map<std::string, std::vector<double>>& promol_exps,
    const std::size_t index_atom_coords = 0,
    const std::size_t i_atom_start = 0
);

}
#endif //CHEMTOOLS_CUDA_INCLUDE_BASIS_TO_GPU_H_
