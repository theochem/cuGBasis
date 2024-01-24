# Add catch as an interface library
set(CATCH_INCLUDE_DIR .)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Find gbasis_cuda_lib
find_package(gbasis_cuda_lib)
set(TEST_SOURCES
        test_densbased.cu
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
