
#include <iostream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <random>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "../include/iodata.h"
#include "../include/eval_rho.cuh"
#include "../include/eval_rho_grad.cuh"
#include "../include/eval_rho_hess.cuh"
#include "../include/eval_lap.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/basis_to_gpu.cuh"
#include "../include/utils.h"

namespace py = pybind11;
using namespace py::literals;


int main() {
  printf("Generate Grid \n");
  // Generate random grid.
  py::scoped_interpreter guard{};

//  std::vector<int> a = {10, 25, 50, 75, 100, 150, 175};
//  for(int i = 0; i < 7; i++){
  int numb_pts = std::pow(100, 3) ; // 10 million points
  std::vector<double> points(3 * numb_pts);
  std::random_device rnd_device;
  std::mt19937  merseene_engine {rnd_device()};
  std::uniform_real_distribution<double> dist {-10, 10};
  auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
  std::generate(points.begin(), points.end(), gen);
  printf("Number of points %d \n", numb_pts);

//  chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk("/home/ali-tehrani/SoftwareProjects/cuGBasis/examples/PHE_TRP_peptide_uwb97xd_def2svpd.fchk");
    chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk("/home/ali-tehrani/SoftwareProjects/cuGBasis/benchmark/DUTLAF10.fchk");
//    chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk("/home/ali-tehrani/SoftwareProjects/cuGBasis/tests/data/qm9_000092_HF_cc-pVDZ.fchk");

  // Evaluate electron density on the cube
//  printf("Evaluate density \n");
  //  // Set up CUDA events for timing
//  cudaEvent_t start, stop;
//  cudaEventCreate(&start);
//  cudaEventCreate(&stop);
//
//  // Start timing
//  cudaEventRecord(start);
//  std::vector<double> result = chemtools::evaluate_electron_density_on_any_grid(iodata, points.data(), numb_pts);
//  // Stop timing
//  cudaEventRecord(stop);
//  cudaEventSynchronize(stop);
//
//  // Calculate elapsed time
//  float elapsedTime;
//  cudaEventElapsedTime(&elapsedTime, start, stop);
//  std::cout << "Time taken to compute density: " << elapsedTime << " ms" << std::endl;





//  //  // Set up CUDA events for timing
//  cudaEvent_t start2, stop2;
//  cudaEventCreate(&start2);
//  cudaEventCreate(&stop2);
//
//  // Start timing
//  cudaEventRecord(start2);
//  std::vector<double> result2 = chemtools::evaluate_electron_density_gradient(iodata, points.data(), numb_pts);
//  // Stop timing
//  cudaEventRecord(stop2);
//  cudaEventSynchronize(stop2);
//
//  // Calculate elapsed time
//  float elapsedTime2;
//  cudaEventElapsedTime(&elapsedTime2, start2, stop2);
//  std::cout << "Time taken to compute gradient: " << elapsedTime2 / 1000 << " s" << std::endl;
//
//
//
//  //  // Set up CUDA events for timing
//  cudaEvent_t start22, stop22;
//  cudaEventCreate(&start22);
//  cudaEventCreate(&stop22);
//
//  // Start timing
//  cudaEventRecord(start22);
//    std::vector<double> result = chemtools::evaluate_electron_density_hessian(iodata, points.data(), numb_pts);
//  // Stop timing
//  cudaEventRecord(stop22);
//  cudaEventSynchronize(stop22);
//
//  // Calculate elapsed time
//  float elapsedTime22;
//  cudaEventElapsedTime(&elapsedTime22, start22, stop22);
//  std::cout << "Time taken to compute hessian: " << elapsedTime22 / 1000 << " s" << std::endl;
    
    
    cudaEvent_t start22, stop22;
    cudaEventCreate(&start22);
    cudaEventCreate(&stop22);
    
    // Start timing
    cudaEventRecord(start22);
    std::vector<double> result = chemtools::evaluate_laplacian(iodata, points.data(), numb_pts);
    // Stop timing
    cudaEventRecord(stop22);
    cudaEventSynchronize(stop22);
    
    // Calculate elapsed time
    float elapsedTime22;
    cudaEventElapsedTime(&elapsedTime22, start22, stop22);
    std::cout << "Time taken to compute Laplacian: " << elapsedTime22 / 1000 << " s" << std::endl;
}

//int main(){
//  printf("Generate Grid \n");
//  // Generate random grid.
//  int numb_pts = std::pow(10, 3); // 10 million points
//  std::vector<double> points(3 * numb_pts);
//  std::random_device rnd_device;
//  std::mt19937  merseene_engine {rnd_device()};
//  std::uniform_real_distribution<double> dist {-5, 5};
//  auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
//  std::generate(points.begin(), points.end(), gen);
//
//  py::scoped_interpreter guard{};
//
//  // Molecular basis set
//  chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk("/home/ali-tehrani/SoftwareProjects/cuGBasis/examples/PHE_TRP_peptide_uwb97xd_def2svpd.fchk");
//  // Get the molecular basis from iodata and put it in constant memory of the gpu.
//  printf("Done iodata \n");
//  chemtools::MolecularBasis molecular_basis = iodata.GetOrbitalBasis();
//  //chemtools::add_mol_basis_to_constant_memory_array(molecular_basis, false, false);
//  int nbasisfuncs = molecular_basis.numb_basis_functions();
//
//  // Put it into constant memory
//  printf("Put in constant memory \n");
//  chemtools::add_mol_basis_to_constant_memory_array(molecular_basis);
//
//  printf("Hello \n");
//  double *d_points;
//  chemtools::cuda_check_errors(cudaMalloc((double **) &d_points, sizeof(double) * 3 * numb_pts));
//  chemtools::cuda_check_errors(cudaMemcpy(d_points,
//                                          points.data(),
//                                          sizeof(double) * 3 * numb_pts,
//                                          cudaMemcpyHostToDevice));
//  printf("Done points \n");
//  cudaDeviceSynchronize();
//  // Function pointers and copy from device to host
//  printf("Transfer to cosntant \n");
//  d_func_t h_contractions_func;
//  cudaMemcpyFromSymbol(&h_contractions_func, chemtools::p_evaluate_contractions, sizeof(d_func_t));
////  printf("Done \n");
//  cudaDeviceSynchronize();
//
//  printf("Allocate Contractions \n");
//  double *d_contractions;
//  chemtools::cuda_check_errors(cudaMalloc((double **) &d_contractions, sizeof(double) * numb_pts * nbasisfuncs));
//  chemtools::cuda_check_errors(cudaMemset(d_contractions, 0, sizeof(double) * numb_pts * nbasisfuncs));
//  cudaDeviceSynchronize();
//
//  // Set up CUDA events for timing
//  cudaEvent_t start, stop;
//  cudaEventCreate(&start);
//  cudaEventCreate(&stop);
//
//  // Start timing
//  cudaEventRecord(start);
//
//
//  int ilen = 320;  // 128 320 1024
//  dim3 threadsPerBlock(ilen);
//  dim3 grid((numb_pts + threadsPerBlock.x - 1) / (threadsPerBlock.x));
////  myKernel<<<grid, threadsPerBlock>>>(d_contractions, d_points, numb_pts, nbasisfuncs, 0);
//  myKernel<<<grid, threadsPerBlock>>>(h_contractions_func, d_contractions, d_points, numb_pts, nbasisfuncs, 0);
//
////  printf("Start Contractions \n");
////  chemtools::evaluate_contractions_from_constant_memory_on_any_grid<<<grid, threadsPerBlock>>>(d_contractions, d_points, numb_pts, nbasisfuncs,
////                                                         0);
//
//  // Stop timing
//  cudaEventRecord(stop);
//  cudaEventSynchronize(stop);
//
//  // Calculate elapsed time
//  float elapsedTime;
//  cudaEventElapsedTime(&elapsedTime, start, stop);
//  std::cout << "Time taken: " << elapsedTime << " ms" << std::endl;
//
//  cudaFree(d_points);
//  cudaDeviceSynchronize();
//  chemtools::print_all<<<1, 1>>>(d_contractions, 10);
//  cudaDeviceSynchronize();
//  printf("Done \n");
//
//  std::vector<double> output_cont(numb_pts * nbasisfuncs);
//  chemtools::cuda_check_errors(cudaMemcpy(output_cont.data(),
//                                          d_contractions,
//                                          sizeof(double) * numb_pts * nbasisfuncs,
//                                          cudaMemcpyDeviceToHost));
//  printf("Output  %lf   %lf   %lf  %lf \n", output_cont[0], output_cont[1], output_cont[2], output_cont[3]);
//  cudaFree(d_contractions);
//
//
//  return 0;
//}
