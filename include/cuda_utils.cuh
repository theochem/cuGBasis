#ifndef CHEMTOOLS_CUDA_INCLUDE_CUDA_UTILS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_CUDA_UTILS_CUH_

#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define CUDART_PI_D 3.141592653589793238462643383279502884197169
#define CUDA_CHECK(err) cuda_check_errors(err, __FILE__, __LINE__)
#define CUBLAS_CHECK(err) cublas_check_errors(err, __FILE__, __LINE__)

namespace chemtools {

/// Copy arrays
__global__ void copy_arrays(double* d_output, double* d_input, int numb_elements);

/// Set up the identity matrix in row major.
__global__ void set_identity_row_major(double* d_array, int nrows, int ncols);

/// assumes both matrices are in the same order. The second one has an output.
__global__ void hadamard_product(
    double* __restrict__ d_array1,
    const double* __restrict__ d_array2,
    int numb_row,
    int numb_col
);

/// same as above but it isn't inplace
__global__ void hadamard_product_outplace(
          double* __restrict__ d_output,
    const double* __restrict__ d_array1,
    const double* __restrict__ d_array2,
    int size);
__global__ void hadamard_product_with_vector_along_row_inplace_for_trapezoidal(double* d_array, const double* d_vec, int numb_row, int numb_col);

/// Hadamard product a tensor (D, M, N) against (M, N) and multiply by two, used by gradient
template <int D>
__global__ void hadamard_product_tensor_with_mat_with_multiply_by_two(
    double* __restrict__ da1,
    const double* __restrict__ da2,
    int size_da2)
{
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_index < size_da2) {
        #pragma unroll D
        for (int i = 0; i < D; i++) {
            da1[global_index + i * size_da2] *= 2.0 * da2[global_index];
        }
    }
}

/// Square-root of each element of the array
__global__ void square_root(double* d_array, int numb_elements);

/// square each element of the array
__global__ void pow_inplace(double* d_array, double power, int numb_elements);

/// Square-root of each element of the array
__global__ void square_root(double* d_array, int numb_elements);

/// square each element of the array
__global__ void pow_inplace(double* d_array, double power, int numb_elements);

/// Multiply array by a scalar in-place
__global__ void multiply_scalar(double* d_array, double scalar, int numb_elements);

/// Convert from C-style to F-style (column-major order) via matrix transpose using cublas
__host__ void array_transpose(cublasHandle_t& handle, double* d_output_col, double* d_input_row, const int numb_rows, const int numb_cols);

/// Matrix row sum with cublas, this is done via \alpha A x, where x is a vector of all ones, A is in Fortran order.
/// order is either "row" or "col".
__host__ double* sum_rows_of_matrix(cublasHandle_t& handle, double* d_matrix, double* d_row_sums,
                                    double* all_ones_ptr, int numb_rows, int numb_cols, const double alpha,
                                    const std::string order = "col");

/// Sum of two matrices to array1
__global__ void sum_two_arrays_inplace(double* d_array1, double* d_array2, int numb_elements);

/// Divide array_1 / array_2 inplace in array_1 with masking, note not serialized correctly
__global__ void divide_inplace(double* d_array1, double* d_array2, int numb_elements, double mask = 1e-16);

/// Printing device memory for debugging.
__global__ void print_first_ten_elements(double* arr);
__global__ void print_all(double* arr, int number);
__global__ void print_firstt_ten_elements(int* arr);
__global__ void print_matrix(double* arr, int row, int column);


/// Checks for cuda errors.
__host__ inline void cuda_check_errors(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    std::cout << "Cuda Error in " << file << ":" << line << \
    "is " << cudaGetErrorString(error) << "( " << error << ") \n" << std::endl;
    throw error;
  }
}

__host__ inline void cuda_check_errors(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cout << "CUDA Error is " << cudaGetErrorString(error) << "( " << error << ") \n" << std::endl;
        throw error;
    }
}
/// Checks for cuBLAS API errors
__host__ static const char* cublas_check_errors(cublasStatus_t error, const char* file, int line)
{
    if (error != CUBLAS_STATUS_SUCCESS)
        std::cout << "cuBLAS error in " << file << ":" << line << "\n Status code: " << error << "\n";
}

__host__ static const char* cublas_check_errors(cublasStatus_t error)
{
    if (error != CUBLAS_STATUS_SUCCESS)
        std::cout << "cuBLAS error in " << "\n Status code: " << error << "\n";
}

}
#endif //CHEMTOOLS_CUDA_INCLUDE_CUDA_UTILS_CUH_
