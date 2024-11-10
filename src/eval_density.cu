#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cstdio>

#include "cublas_v2.h"

#include "eval_density.cuh"
#include "cuda_utils.cuh"
#include "cuda_basis_utils.cuh"
#include "basis_to_gpu.cuh"
#include "run.cuh"

using namespace chemtools;

__device__ d_func_t chemtools::p_eval_AOs = eval_AOs_from_constant_memory_on_any_grid;

__device__ __forceinline__ void chemtools::eval_AOs(
          double*  d_AO_vals,
    const double3& pt,
    const int&     n_pts,
          uint&    idx,
    const int&     iorb_start)
{
    uint ibasis     = 0;                                       // Index to go over constant memory.
    uint iorb       = iorb_start;                              // Index to go over rows of d_AO_vals
    uint n_cshells  = (uint) g_constant_basis[ibasis++];       // Number Contraction Shells
    double4 AO1 = {0.0, 0.0, 0.0, 0.0};                        // Intermediate values evaluates
    double4 AO2 = {0.0, 0.0, 0.0, 0.0};                        // atomic orbitals/contractions for
    double4 AO3 = {0.0, 0.0, 0.0, 0.0};                        // angmom/shell
    double3 AO4 = {0.0, 0.0, 0.0};
    
    #pragma unroll 1
    for(int ishell = 0; ishell < n_cshells; ishell++) {
        double3 r_A = {
            pt.x - g_constant_basis[ibasis++],
            pt.y - g_constant_basis[ibasis++],
            pt.z - g_constant_basis[ibasis++]
        };
        uint n_seg_shells  = (uint) g_constant_basis[ibasis++];   // Number Segmented shell
        uint n_prims       = (uint) g_constant_basis[ibasis++];   // Number of primitives inside shell
        #pragma unroll 1
        for(int iseg=0; iseg < n_seg_shells; iseg++) {
            int L = (int) g_constant_basis[ibasis + n_prims + (n_prims + 1) * iseg];
            if (L == S_TYPE)
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a  = g_constant_basis[ibasis + i_prim];
                    double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    AO1.x += normalization_primitive_s(a) * ce;
                }
            else if (L == P_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a  = g_constant_basis[ibasis + i_prim];
                    double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    double norm = normalization_primitive_p(a);
                    AO1.x += norm * r_A.x * ce;
                    AO1.y += norm * r_A.y * ce;
                    AO1.z += norm * r_A.z * ce;
                }
            } else if (L == D_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a  = g_constant_basis[ibasis + i_prim];
                    double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // The ordering is ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
                    AO1.x += normalization_primitive_d(a, 2, 0, 0) * r_A.x * r_A.x * ce;
                    AO1.y += normalization_primitive_d(a, 0, 2, 0) * r_A.y * r_A.y * ce;
                    AO1.z += normalization_primitive_d(a, 0, 0, 2) * r_A.z * r_A.z * ce;
                    AO1.w += normalization_primitive_d(a, 1, 1, 0) * r_A.x * r_A.y * ce;
                    AO2.x += normalization_primitive_d(a, 1, 0, 1) * r_A.x * r_A.z * ce;
                    AO2.y += normalization_primitive_d(a, 0, 1, 1) * r_A.y * r_A.z * ce;
                }
            } else if (L == DP_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c    = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a    = g_constant_basis[ibasis + i_prim];
                    double ce   = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // Negatives are s denoting sine and c denoting cosine.
                    // Fchk ordering is  ['c0', 'c1', 's1', 'c2', 's2']
                    double norm_const = normalization_primitive_pure_d(a);
                    AO1.x += norm_const * solid_harmonic_d(0, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.y += norm_const * solid_harmonic_d(1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.z += norm_const * solid_harmonic_d(-1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.w += norm_const * solid_harmonic_d(2, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.x += norm_const * solid_harmonic_d(-2, r_A.x, r_A.y, r_A.z) * ce;
                }
            } else if (L == F_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];   // Coeff
                    double a  = g_constant_basis[ibasis + i_prim];                                     // Expon
                    double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // The ordering is ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
                    AO1.x += normalization_primitive_f(a, 3, 0, 0) * r_A.x * r_A.x * r_A.x * ce;
                    AO1.y += normalization_primitive_f(a, 0, 3, 0) * r_A.y * r_A.y * r_A.y * ce;
                    AO1.z += normalization_primitive_f(a, 0, 0, 3) *r_A.z * r_A.z * r_A.z *ce;
                    AO1.w += normalization_primitive_f(a, 1, 2, 0) *r_A.x * r_A.y * r_A.y *ce;
                    AO2.x += normalization_primitive_f(a, 2, 1, 0) *r_A.x * r_A.x * r_A.y *ce;
                    AO2.y += normalization_primitive_f(a, 2, 0, 1) *r_A.x * r_A.x * r_A.z *ce;
                    AO2.z += normalization_primitive_f(a, 1, 0, 2) *r_A.x * r_A.z * r_A.z *ce;
                    AO2.w += normalization_primitive_f(a, 0, 1, 2) *r_A.y * r_A.z * r_A.z *ce;
                    AO3.x += normalization_primitive_f(a, 0, 2, 1) *r_A.y * r_A.y * r_A.z *ce;
                    AO3.y += normalization_primitive_f(a, 1, 1, 1) *r_A.x * r_A.y * r_A.z *ce;
                }
            } else if (L == SF_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c          = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];   // Coeff
                    double a          = g_constant_basis[ibasis + i_prim];                                     // Expon
                    double ce         = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']
                    double norm_const = normalization_primitive_pure_f(a);
                    AO1.x += norm_const * solid_harmonic_f(0, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.y += norm_const * solid_harmonic_f(1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.z += norm_const * solid_harmonic_f(-1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.w += norm_const * solid_harmonic_f(2, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.x += norm_const * solid_harmonic_f(-2, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.y += norm_const * solid_harmonic_f(3, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.z += norm_const * solid_harmonic_f(-3, r_A.x, r_A.y, r_A.z) * ce;
                }
            } else if (L == G_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c  = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a  = g_constant_basis[ibasis + i_prim];
                    double ce = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // The ordering is ['zzzz', 'yzzz', 'yyzz', 'yyyz', 'yyyy', 'xzzz', 'xyzz', 'xyyz', 'xyyy', 'xxzz',
                    //                  'xxyz', 'xxyy', 'xxxz', 'xxxy', 'xxxx']
                    AO1.x += normalization_primitive_g(a, 0, 0, 4) * r_A.z * r_A.z * r_A.z * r_A.z * ce;
                    AO1.y += normalization_primitive_g(a, 0, 1, 3) * r_A.y * r_A.z * r_A.z * r_A.z * ce;
                    AO1.z += normalization_primitive_g(a, 0, 2, 2) * r_A.y * r_A.y * r_A.z * r_A.z * ce;
                    AO1.w += normalization_primitive_g(a, 0, 3, 1) * r_A.y * r_A.y * r_A.y * r_A.z * ce;
                    AO2.x += normalization_primitive_g(a, 0, 4, 0) * r_A.y * r_A.y * r_A.y * r_A.y * ce;
                    AO2.y += normalization_primitive_g(a, 1, 0, 3) * r_A.x * r_A.z * r_A.z * r_A.z * ce;
                    AO2.z += normalization_primitive_g(a, 1, 1, 2) * r_A.x * r_A.y * r_A.z * r_A.z * ce;
                    AO2.w += normalization_primitive_g(a, 1, 2, 1) * r_A.x * r_A.y * r_A.y * r_A.z * ce;
                    AO3.x += normalization_primitive_g(a, 1, 3, 0) * r_A.x * r_A.y * r_A.y * r_A.y * ce;
                    AO3.y += normalization_primitive_g(a, 2, 0, 2) * r_A.x * r_A.x * r_A.z * r_A.z * ce;
                    AO3.z += normalization_primitive_g(a, 2, 1, 1) * r_A.x * r_A.x * r_A.y * r_A.z * ce;
                    AO3.w += normalization_primitive_g(a, 2, 2, 0) * r_A.x * r_A.x * r_A.y * r_A.y * ce;
                    AO4.x += normalization_primitive_g(a, 3, 0, 1) * r_A.x * r_A.x * r_A.x * r_A.z * ce;
                    AO4.y += normalization_primitive_g(a, 3, 1, 0) * r_A.x * r_A.x * r_A.x * r_A.y * ce;
                    AO4.z += normalization_primitive_g(a, 4, 0, 0) * r_A.x * r_A.x * r_A.x * r_A.x * ce;
                }
            } else if (L == SG_TYPE) {
                #pragma unroll 1
                for (int i_prim = 0; i_prim < n_prims; i_prim++) {
                    double c   = g_constant_basis[ibasis + n_prims * (iseg + 1) + i_prim + 1 + iseg];
                    double a   = g_constant_basis[ibasis + i_prim];
                    double ce  = c * exp(-a * (r_A.x * r_A.x + r_A.y * r_A.y + r_A.z * r_A.z));
                    // ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3', 'c4', 's4']
                    double norm_const = normalization_primitive_pure_g(a);
                    AO1.x += norm_const * solid_harmonic_g(0, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.y += norm_const * solid_harmonic_g(1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.z += norm_const * solid_harmonic_g(-1, r_A.x, r_A.y, r_A.z) * ce;
                    AO1.w += norm_const * solid_harmonic_g(2, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.x += norm_const * solid_harmonic_g(-2, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.y += norm_const * solid_harmonic_g(3, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.z += norm_const * solid_harmonic_g(-3, r_A.x, r_A.y, r_A.z) * ce;
                    AO2.w += norm_const * solid_harmonic_g(4, r_A.x, r_A.y, r_A.z) * ce;
                    AO3.x += norm_const * solid_harmonic_g(-4, r_A.x, r_A.y, r_A.z) * ce;
                }
            }// End angmom.
            
            // Update index that goes over each contraction and store AO_evals in global memory.
            if (L == S_TYPE) {
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                iorb += 1;
            }
            else if (L == P_TYPE) {
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                iorb += 3;
            }
            else if (L == D_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                d_AO_vals[idx + (iorb + 5) * n_pts] = AO2.y;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                iorb += 6;
            }
            else if (L == DP_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                iorb += 5;
            }
            else if (L == F_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                d_AO_vals[idx + (iorb + 5) * n_pts] = AO2.y;
                d_AO_vals[idx + (iorb + 6) * n_pts] = AO2.z;
                d_AO_vals[idx + (iorb + 7) * n_pts] = AO2.w;
                d_AO_vals[idx + (iorb + 8) * n_pts] = AO3.x;
                d_AO_vals[idx + (iorb + 9) * n_pts] = AO3.y;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                AO3 = {0.0, 0.0, 0.0, 0.0};
                iorb += 10;
            }
            else if (L == SF_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                d_AO_vals[idx + (iorb + 5) * n_pts] = AO2.y;
                d_AO_vals[idx + (iorb + 6) * n_pts] = AO2.z;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                iorb += 7;
            }
            else if (L == G_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                d_AO_vals[idx + (iorb + 5) * n_pts] = AO2.y;
                d_AO_vals[idx + (iorb + 6) * n_pts] = AO2.z;
                d_AO_vals[idx + (iorb + 7) * n_pts] = AO2.w;
                d_AO_vals[idx + (iorb + 8) * n_pts] = AO3.x;
                d_AO_vals[idx + (iorb + 9) * n_pts] = AO3.y;
                d_AO_vals[idx + (iorb + 10) * n_pts] = AO3.z;
                d_AO_vals[idx + (iorb + 11) * n_pts] = AO3.w;
                d_AO_vals[idx + (iorb + 12) * n_pts] = AO4.x;
                d_AO_vals[idx + (iorb + 13) * n_pts] = AO4.y;
                d_AO_vals[idx + (iorb + 14) * n_pts] = AO4.z;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                AO3 = {0.0, 0.0, 0.0, 0.0};
                AO4 = {0.0, 0.0, 0.0};
                iorb += 15;
            }
            else if (L == SG_TYPE){
                d_AO_vals[idx + iorb * n_pts] = AO1.x;
                d_AO_vals[idx + (iorb + 1) * n_pts] = AO1.y;
                d_AO_vals[idx + (iorb + 2) * n_pts] = AO1.z;
                d_AO_vals[idx + (iorb + 3) * n_pts] = AO1.w;
                d_AO_vals[idx + (iorb + 4) * n_pts] = AO2.x;
                d_AO_vals[idx + (iorb + 5) * n_pts] = AO2.y;
                d_AO_vals[idx + (iorb + 6) * n_pts] = AO2.z;
                d_AO_vals[idx + (iorb + 7) * n_pts] = AO2.w;
                d_AO_vals[idx + (iorb + 8) * n_pts] = AO3.x;
                AO2 = {0.0, 0.0, 0.0, 0.0};
                AO3 = {0.0, 0.0, 0.0, 0.0};
                iorb += 9;
            }
            AO1 = {0.0, 0.0, 0.0, 0.0};
        //} // End updating segmented shell.
        } // End going over contractions/primitives of a single segmented shell.
    // Update index of constant memory, add the number of exponents then number of angular momentum terms then
    //        add the number of coefficients.
    ibasis += n_prims + (n_seg_shells + n_seg_shells * n_prims);
    } // End Basis
}


__global__ __launch_bounds__(128) void chemtools::eval_AOs_from_constant_memory_on_any_grid(
          double* __restrict__ d_AO_vals,
    const double* __restrict__ d_points,
    const int     n_pts,
    const int     n_cshells,
    const int     iorb_start
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pts) {
        // Get the grid points where `d_points` is in column-major order with shape (N, 3)
        double pt_x = d_points[idx];
        double pt_y = d_points[idx + n_pts];
        double pt_z = d_points[idx + n_pts * 2];
        // Evaluate the contractions and store it in d_AO_vals
        chemtools::eval_AOs(d_AO_vals, {pt_x, pt_y, pt_z}, n_pts, idx, iorb_start);
    }
}



__host__ std::vector<double> chemtools::evaluate_electron_density_on_any_grid(
    IOData& iodata, const double* h_points, const int n_pts)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    std::vector<double> density = evaluate_electron_density_on_any_grid_handle(
      handle, iodata, h_points, n_pts
    );
    cublasDestroy(handle); // cublas handle is no longer needed infact most of
    return density;
}


__host__ std::vector<double> chemtools::evaluate_electron_density_on_any_grid_handle(
          cublasHandle_t& handle,
          IOData&         iodata,
    const double*         h_points,
    const int             n_pts
) {
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
    
    // Resulting electron density
    std::vector<double> h_rho(n_pts);
    
    // Iterate through each chunk of the data set.
    size_t index_to_copy = 0;  // Index on where to start copying to h_electron_density (start of sub-grid)
    size_t i_iter        = 0;
    while(index_to_copy < n_pts) {
        // Calculate number of points to do
        size_t npts_iter    = std::min(
            n_pts - i_iter * chunks.pts_per_iter,
            chunks.pts_per_iter
        );
        printf("Numbe rof points per iter %zu %zu \n", npts_iter, chunks.pts_per_iter);
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
            d_AOs,
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
        
        // Temp array matrix multiplication of one_rdm and contraction array.
        double *d_temp;
        CUDA_CHECK(cudaMalloc((double **) &d_temp, AO_data_size));
        
        // Matrix mult. of one rdm with the contractions array. Everything is in row major order.
        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            npts_iter, iodata.GetOneRdmShape(), iodata.GetOneRdmShape(),
            &alpha,
            d_AOs, npts_iter,
            d_one_rdm, iodata.GetOneRdmShape(),
            &beta,
            d_temp, npts_iter
        ));
        cudaFree(d_one_rdm);
        
        // Do a hadamard product between d_temp and d_contraction and store it in d_temp
        constexpr int HADAMARD_THREADS = 320;
        dim3 hadamard_threads(HADAMARD_THREADS);
        dim3 hadamard_blocks(
            (npts_iter * n_basis + HADAMARD_THREADS - 1) / HADAMARD_THREADS
        );
        hadamard_product<<<hadamard_blocks, hadamard_threads>>>(
            d_temp, d_AOs, n_basis, npts_iter
        );
        cudaFree(d_AOs);
        
        // Allocate device memory for electron density.
        double *d_electron_density;
        CUDA_CHECK(cudaMalloc((double **) &d_electron_density, npts_iter * sizeof(double)));
        
        // Sum columns using ones vector. Using d_temp is in row major order.
        thrust::device_vector<double> all_ones(sizeof(double) * n_basis, 1.0);
        double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());
        CUBLAS_CHECK(cublasDgemv(
            handle, CUBLAS_OP_N,
            npts_iter, n_basis, &alpha, d_temp, npts_iter, deviceVecPtr, 1, &beta, d_electron_density, 1
        ));
        cudaFree(d_temp);
        
        // Transfer electron density from device memory to host memory.
        CUDA_CHECK(cudaMemcpy(
            &h_rho[0] + index_to_copy,
            d_electron_density,
            npts_iter * sizeof(double), cudaMemcpyDeviceToHost
        ));
        cudaFree(d_electron_density);
        
        // Update lower-bound of the grid for the next iteration
        index_to_copy += npts_iter;
        i_iter++;
    }
    return h_rho;
}

