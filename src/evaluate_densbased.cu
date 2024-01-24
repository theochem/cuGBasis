
#include "../include/evaluate_densbased.cuh"
#include "../include/cuda_utils.cuh"

__host__ std::vector<double> gbasis::compute_norm_of_3d_vector(double *h_points, const int knumb_pts){
  std::vector<double> h_norm(knumb_pts);
  // Assumes h_points is in column-order.

  // Transfer to GPU memory
  double *d_points;
  gbasis::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * knumb_pts));
  gbasis::cuda_check_errors(cudaMemcpy(d_points, h_points,
                                       sizeof(double) * 3 * knumb_pts,
                                       cudaMemcpyHostToDevice));

  // Square the GPU memory
  dim3 threadsPerBlock(320);
  dim3 grid((3 * knumb_pts + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::pow_inplace<<<grid, threadsPerBlock>>>(d_points, 2.0, 3 * knumb_pts);

  // Sum the first row with the second row
  dim3 grid2((knumb_pts + threadsPerBlock.x - 1) / (threadsPerBlock.x));
  gbasis::sum_two_arrays_inplace<<<grid2, threadsPerBlock>>>(d_points, &d_points[knumb_pts], knumb_pts);

  // Add the third row
  gbasis::sum_two_arrays_inplace<<<grid2, threadsPerBlock>>>(&d_points[0], &d_points[2 * knumb_pts], knumb_pts);

  // Take square root the GPU memory
  gbasis::square_root<<<grid2, threadsPerBlock>>>(d_points, knumb_pts);

  // Transfer from GPU memory to CPU
  gbasis::cuda_check_errors(cudaMemcpy(h_norm.data(), d_points, sizeof(double) * knumb_pts, cudaMemcpyDeviceToHost));
  cudaFree(d_points);

  return h_norm;
}
