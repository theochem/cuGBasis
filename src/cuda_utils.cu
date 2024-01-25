#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

#include "../include/cuda_utils.cuh"

#include <thrust/device_vector.h>


__global__ void chemtools::set_identity_row_major(double* d_array, int nrows, int ncols){
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if(x < nrows && y < ncols) {
    if(x == y)
      d_array[x + nrows * y] = 1.;
    else
      d_array[x + nrows * y] = 0.;
  }
}

__global__ void chemtools::hadamard_product(double* d_array1, double* d_array2, int numb_row, int numb_col) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_row * numb_col) {
    d_array1[global_index] *= d_array2[global_index];
  }
}

__global__ void chemtools::hadamard_product_outplace(double* d_output, double* d_array1, double* d_array2, int numb_row, int numb_col) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_row * numb_col) {
    d_output[global_index] = d_array1[global_index] * d_array2[global_index];
  }
}

__global__ void chemtools::multiply_scalar(double* d_array, double scalar, int numb_elements) {
  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    d_array[global_index] *= scalar;
  }
}

__global__ void chemtools::hadamard_product_with_vector_along_row_inplace_for_trapezoidal(double* d_array, const double* d_vec, int numb_row, int numb_col) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  // Maybe it's better to store d_vec elements into shared memory, as to avoid always going to global memory.
  if (ix < numb_col * numb_row) {
    int col_index = ix % numb_col;
    d_array[ix] *= d_vec[col_index];
    if (col_index == 0 || col_index == numb_col - 1) {
      d_array[ix] /= 2.0;
    }
  }
}

__global__ void chemtools::square_root(double* d_array, int numb_elements) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    d_array[global_index] = std::sqrt(d_array[global_index]);
  }
}

__global__ void chemtools::pow_inplace(double* d_array, double power, int numb_elements) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    d_array[global_index] = std::pow(d_array[global_index], power);
    // d_array[global_index] = d_array[global_index] * d_array[global_index];
  }
}

__global__ void chemtools::divide_inplace(
    double* d_array1, double* d_array2, const int numb_elements, const double mask
) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    // Adding this if-statements, stops the threads and is not proper GPU coding practice
    if(d_array2[global_index] > mask){
      d_array1[global_index] = d_array1[global_index] / d_array2[global_index];
    }
    else {
      d_array1[global_index] = mask;
    }
  }
}

__global__ void chemtools::copy_arrays(double* d_output, double* d_input, int numb_elements) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    d_output[global_index] = d_input[global_index];
  }
}

__host__ void chemtools::array_transpose(
    cublasHandle_t& handle, double* d_output_col, double* d_input_row, const int numb_rows, const int numb_cols) {
  double const alpha(1.0);
  double const beta(0.0);
  chemtools::cublas_check_errors(
      cublasDgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, numb_rows, numb_cols, &alpha, d_input_row, numb_cols, &beta, d_input_row,
                   numb_rows, d_output_col, numb_rows )
  );
}



__host__ double* chemtools::sum_rows_of_matrix(cublasHandle_t& handle, double* d_matrix, double* d_row_sums,
                                            double* all_ones_ptr, int numb_rows, int numb_cols, const double alpha,
                                            const std::string order) {
  double beta = 0.;
  // I moved these to the outside to same time when computing the kernel.
//  double* d_row_sums;
//  cudaMalloc((int **) &d_row_sums, sizeof(double) * numb_rows);

//  thrust::device_vector<double> all_ones(sizeof(double) * numb_cols, 1.0);
//  double *deviceVecPtr = thrust::raw_pointer_cast(all_ones.data());

  if (order == "row"){
    chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_T, numb_cols, numb_rows,
                                            &alpha, d_matrix, numb_cols, all_ones_ptr, 1, &beta,
                                            d_row_sums, 1));
  }
  else if (order == "col") {
    chemtools::cublas_check_errors(cublasDgemv(handle, CUBLAS_OP_N, numb_rows, numb_cols,
                                            &alpha, d_matrix, numb_rows, all_ones_ptr, 1, &beta,
                                            d_row_sums, 1));
  }

  return d_row_sums;
}


__global__ void chemtools::sum_two_arrays_inplace(double* d_array1, double* d_array2, int numb_elements) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(global_index < numb_elements) {
    d_array1[global_index] = d_array1[global_index] + d_array2[global_index];
  }
}


__global__ void chemtools::print_first_ten_elements(double* arr) {
  printf("%E %E %E %E  %E %E \n", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
}
__global__ void chemtools::print_all(double* arr, int number) {
  for(int i =0; i < number; i++) {
    printf("%.12E    ", arr[i]);
  }
}

__global__ void chemtools::print_matrix(double* arr, int row, int column) {\
  int k = 0;
  for(int i =0; i < row; i++) {
    printf("ith row %d \n", i);
    for (int j = 0; j < column; j++) {
      printf("%E    ", arr[k]);
      k += 1;
    }
    printf("\n\n\n");
  }
}

__global__ void chemtools::print_firstt_ten_elements(int* arr) {
  printf("%d %d %d %d %d %d \n", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
}


