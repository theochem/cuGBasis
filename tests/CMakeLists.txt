
# Add catch as an interface library
set(CATCH_INCLUDE_DIR .)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Pybind11 linking to the correct python
#set(PYBIND11_FINDPYTHON ON)
#SET(PYBIND11_PYTHON_VERSION 3.7)
#set(Python_ROOT_DIR /home/ali-tehrani/miniconda2/envs/py37/)
#find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
#message("Python path is " ${Python_EXECUTABLE})
#message("Found numpy " ${Python3_NumPy_FOUND})
#add_subdirectory(../libs/pybind11)

#add_subdirectory(./Catch2)


# Find gbasis_cuda_lib
find_package(gbasis_cuda_lib)

#add_library(test_lib STATIC  ./catch.hpp)
#target_link_libraries(test_lib PRIVATE pybind11::embed) # Used to be able to do python
#target_link_libraries(test_lib PRIVATE gbasis_cuda_lib)

set(TEST_SOURCES
        test_gradient.cu
        test_kinetic_density.cu
        test_laplacian.cu
        test_electron_density.cu
        test_electrostatic_potential.cu
        )
# Add test executable
add_executable (tests main_test.cu catch.hpp ${TEST_SOURCES})
target_link_libraries(tests PRIVATE pybind11::embed)
target_link_libraries(tests PRIVATE gbasis_cuda_lib)
#target_link_libraries(tests PRIVATE Catch2::Catch2)
