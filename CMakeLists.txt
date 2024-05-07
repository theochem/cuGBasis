cmake_minimum_required(VERSION 3.16)
project(cugbasis)
enable_language(CXX)
# TODO: Give command line option for CMAKE to find nvcc!
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)

# Add Eigen header files
find_package(Eigen3 3.3 QUIET)

# If user wants to add the path to EIGEN themselves then they should do something like so:
if( Eigen3_FOUND )
	message("Eigen was found")
else()
	message("Eigen was not found: Automatically try to install")
	add_subdirectory(libs/eigen)
endif()


# Add pybind11 header files
# Finding Python Needed to do this for CLION project for my own computer.
set(PYBIND11_FINDPYTHON ON)
SET(PYBIND11_PYTHON_VERSION 3.7)
# TODO: GIve a command-line option for CMAKE to find Python
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
add_subdirectory(./libs/pybind11)

# Create a CUDA library chemtools_cuda using nvcc compiler that codes the entire CUDA/C++.
set(SOURCES
	src/basis_to_gpu.cu
#	src/boys_functions.cu
	src/iodata.cpp
	src/cuda_basis_utils.cu
	src/cuda_utils.cu
	src/integral_coeffs.cu
	src/evaluate_density.cu
	src/evaluate_gradient.cu
	src/evaluate_electrostatic.cu
	src/utils.cpp
	src/evaluate_laplacian.cu
	src/evaluate_kinetic_dens.cu
	src/evaluate_densbased.cu
	src/evaluate_hessian.cu
	src/pymolecule.cu
)
set(HEADERS
	include/basis_to_gpu.cuh
	include/contracted_shell.h
	include/iodata.h
	include/boys_functions.cuh
	include/cuda_basis_utils.cuh
	include/cuda_utils.cuh
	include/integral_coeffs.cuh
	include/evaluate_density.cuh
	include/evaluate_gradient.cuh
	include/evaluate_electrostatic.cuh
	include/utils.h
	include/evaluate_laplacian.cuh
	include/evaluate_kinetic_dens.cuh
	include/evaluate_densbased.cuh
	include/evaluate_hessian.cuh
	include/pymolecule.cuh
)
find_package(CUDAToolkit REQUIRED)  # The point of this is to call CUDA code from C++ but without including CUDA code.
                                    #  see stackexchange - "linking of cuda library in cmake
#set_source_files_properties(
#  ./src/integral_coeffs.cu
#  PROPERTIES
#  COMPILE_FLAGS "-G -g"
#)
#set_source_files_properties(
#	./src/evaluate_electrostatic.cu
#	PROPERTIES
#	COMPILE_FLAGS "-G -g"
#	)

add_library(cugbasis_lib SHARED ${HEADERS} ${SOURCES})
set_target_properties(cugbasis_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# If the user didn't define CUDA architecture
if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
	# If cmake version is greater than 3.24, automatically find the right CUDA Architecture
	if("$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
		if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
			set_property(TARGET cugbasis_lib PROPERTY CUDA_ARCHITECTURES native)
		endif()
	else()
		set_property(TARGET cugbasis_lib PROPERTY CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
	endif ()
else()
	# If the user did define the CUDA Archiecture then add it here.
	set_property(TARGET cugbasis_lib PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_compile_options(cugbasis_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -rdc=true --expt-relaxed-constexpr -v
	>) # -rdc=true  std=c++14 isn't needed since CMAKE_CUDA_STANDARD already set Debuging -G -g turns off optimization, -O3 turns on optmization, -gencode=arch=compute_35,code=sm_35
# --expt-relaxed-constexpr is needed for Boys_function.h to work.
target_link_libraries(cugbasis_lib PRIVATE ${CUDA_cudadevrt_LIBRARY})
target_link_libraries(cugbasis_lib PRIVATE CUDA::cublas)
target_link_libraries(cugbasis_lib PRIVATE CUDA::curand)
target_link_libraries(cugbasis_lib PUBLIC pybind11::embed) # Used to call IODATA to C++\
if ( Eigen3_FOUND )
	target_link_libraries(cugbasis_lib PUBLIC Eigen3::Eigen)
else()
	target_link_libraries(cugbasis_lib PUBLIC eigen)
endif()


# Link the chemtools_cuda library to pybind11,
set(${LINK_TO_PYTHON} OFF CACHE BOOL "Link to python")  # This option is set to True in setup.py
if(${LINK_TO_PYTHON})
	pybind11_add_module(cugbasis src/pybind.cpp)
	# cugbasis is compiled as a shared library and then linked via RPATH to Python package via Python_SITEARCH variable
	#    obtained from find_package(Python ...) The other alternative is to use the compiled library that is inside
	#    the build folder, this might be a better approach then the current.
	set_target_properties(cugbasis PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "${Python_SITEARCH}")
	message("SHARED LIBRARY INSTALLED TO Python ${Python_SITEARCH}")
	target_link_libraries(cugbasis PRIVATE cugbasis_lib)
	target_link_libraries(cugbasis PRIVATE CUDA::cublas)
endif()

#add_subdirectory(tests)
# Compilation of the tests, not needed e.g. if `LINK_TO_PYTHON` IS True/ON
set(${DONT_COMPILE_TESTS} OFF CACHE BOOL "Don't compile the tests")
if(NOT ${DONT_COMPILE_TESTS})
	add_subdirectory(tests)
endif()

#add_executable (main_c main.cu)
#target_link_libraries(main_c PRIVATE chemtools_cuda_lib)
#target_link_libraries(main_c PUBLIC pybind11::embed)
