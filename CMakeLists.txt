cmake_minimum_required(VERSION 3.16)
project(gbasis_cuda)
enable_language(CXX)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# TODO: Give command line option for CMAKE to find nvcc!
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)

# Add Eigen header files
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(${EIGEN3_INCLUDE_DIR})
# If user wants to add the path to EIGEN themselves then they should do something like so:
#   TODO: Give a command line option for CMAKE to do so.
#include_directories(Eigen3 /home/ali-tehrani/miniconda2/envs/py37/include/eigen3/)

# Add pybind11 header files
# Finding Python Needed to do this for CLION project for my own computer.
set(PYBIND11_FINDPYTHON ON)
SET(PYBIND11_PYTHON_VERSION 3.7)
# TODO: GIve a command-line option for CMAKE to find Python
#set(Python_ROOT_DIR  /home/ali-tehrani/miniconda2/envs/py37/)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
add_subdirectory(./libs/pybind11)

# Add Thrust
message("Install Thrust")
add_subdirectory(./libs/thrust)
thrust_create_target(Thrust)


# Create a CUDA library gbasis_cuda using nvcc compiler that codes the entire CUDA/C++.
set(SOURCES
        src/basis_to_gpu.cu
        src/boys_functions.cu
        src/iodata.cpp
        src/pymolecule.cu
        src/cuda_basis_utils.cu
        src/cuda_utils.cu
		src/integral_coeffs.cu
        src/evaluate_density.cu
        src/evaluate_gradient.cu
		src/evaluate_electrostatic.cu
		src/utils.cpp
		src/evaluate_laplacian.cu
		src/evaluate_kinetic_dens.cu)
set(HEADERS
        include/basis_to_gpu.cuh
        include/contracted_shell.h
        include/iodata.h
        include/pymolecule.cuh
        include/boys_functions.cuh
        include/cuda_basis_utils.cuh
        include/cuda_utils.cuh
		include/integral_coeffs.cuh
        include/evaluate_density.cuh
        include/evaluate_gradient.cuh
		include/evaluate_electrostatic.cuh
		include/utils.h
        include/evaluate_laplacian.cuh
		include/evaluate_kinetic_dens.cuh)
message("PRINT2 ${CUDA_NVCC_FLAGS}")
add_library(gbasis_cuda_lib SHARED ${HEADERS} ${SOURCES})
set_target_properties(gbasis_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gbasis_cuda_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -rdc=true --expt-relaxed-constexpr -O3
	>) # -rdc=true  std=c++14 isn't needed since CMAKE_CUDA_STANDARD already set Debuging -G -g turns off optimization, -O3 turns on optmization, -gencode=arch=compute_35,code=sm_35
# --expt-relaxed-constexpr is needed for Boys_function.h to work.
target_link_libraries(gbasis_cuda_lib PRIVATE ${CUDA_cudadevrt_LIBRARY})
target_link_libraries(gbasis_cuda_lib PRIVATE -lcublas -lcurand)
target_link_libraries(gbasis_cuda_lib PUBLIC pybind11::embed) # Used to call IODATA to C++\
target_link_libraries(gbasis_cuda_lib PUBLIC Eigen3::Eigen)
target_link_libraries(gbasis_cuda_lib PRIVATE Thrust)

#add_executable(main main.cu)
#target_link_libraries(main PRIVATE gbasis_cuda_lib)
# Link the gbasis_cuda library to pybind11
find_package(CUDAToolkit)  # The point of this is to call CUDA code from C++ but without including CUDA code.
                           #  see stackexchange - "linking of cuda library in cmake"
pybind11_add_module(gbasis_cuda src/pybind.cpp)
target_link_libraries(gbasis_cuda PRIVATE gbasis_cuda_lib)
target_link_libraries(gbasis_cuda PRIVATE CUDA::cublas)


add_subdirectory(tests)
