cmake_minimum_required(VERSION 3.16)
project(cugbasis)
enable_language(CXX)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)


if(POLICY CMP0075)
	cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
	cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0057)
	cmake_policy(SET CMP0057 NEW)
endif()

if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()


find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
	# Update submodules as needed
	option(GIT_SUBMODULE "Check submodules during build" ON)
	if(GIT_SUBMODULE)
		message(STATUS "Submodule update: ${CMAKE_CURRENT_SOURCE_DIR} ")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
				WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
				RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "Cloning git submodules failed with ${GIT_SUBMOD_RESULT}, please checkout submodules manually")
		endif()
	endif()
endif()

# Add Eigen header files
find_package(Eigen3 3.3 QUIET)

# If user wants to add the path to EIGEN themselves then they should do something like so:
if( Eigen3_FOUND )
	message(STATUS "Found Eigen3: ${EIGEN3_VERSION} at ${EIGEN3_INCLUDE_DIR}")
else()
	message("Eigen was not found: Automatically try to install")
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen)
endif()


# Add pybind11 header files
# Finding Python Needed to do this for CLION project for my own computer.
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "Python Version: ${Python_VERSION}")
message(STATUS "Python Include Dir: ${Python_INCLUDE_DIRS}")
message(STATUS "NumPy Include Dir: ${Python_NumPy_INCLUDE_DIRS}")
add_subdirectory(${PROJECT_SOURCE_DIR}/libs/pybind11)



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
	src/evaluate_promolecular.cu
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
	include/evaluate_promolecular.cuh
	include/utils.h
	include/evaluate_laplacian.cuh
	include/evaluate_kinetic_dens.cuh
	include/evaluate_densbased.cuh
	include/evaluate_hessian.cuh
	include/pymolecule.cuh
)
find_package(CUDAToolkit REQUIRED)  # The point of this is to call CUDA code from C++ but without including CUDA code.
                                    #  see stackexchange - "linking of cuda library in cmake
set_source_files_properties(
  ./src/integral_coeffs.cu
  PROPERTIES
  COMPILE_FLAGS "-G -g"
)
set_source_files_properties(
	./src/evaluate_electrostatic.cu
	PROPERTIES
	COMPILE_FLAGS "-G -g"
	)

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
		set_property(TARGET cugbasis_lib PROPERTY CUDA_ARCHITECTURES 52 60 61 75)  # Set a default CUDA_ARCHITECTURE
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
	message("SHARED LIBRARY INSTALLED TO Python ${Python3_SITEARCH}")
	set_target_properties(cugbasis PROPERTIES INSTALL_RPATH "${Python3_SITEARCH}")
	target_link_libraries(cugbasis PRIVATE cugbasis_lib)
	target_link_libraries(cugbasis PRIVATE CUDA::cublas)

	install(TARGETS cugbasis DESTINATION ${Python3_SITEARCH})
	install(TARGETS cugbasis_lib DESTINATION ${Python3_SITEARCH})
endif()

# Compilation of the tests, not needed e.g. if `LINK_TO_PYTHON` IS True/ON
set(${DONT_COMPILE_TESTS} OFF CACHE BOOL "Don't compile the tests")
if(NOT ${DONT_COMPILE_TESTS})
	add_subdirectory(tests)
endif()

#add_executable (main_c main.cu)
#target_link_libraries(main_c PRIVATE chemtools_cuda_lib)
#target_link_libraries(main_c PUBLIC pybind11::embed)
