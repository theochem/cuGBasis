# Add catch as an interface library
set(CATCH_INCLUDE_DIR .)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Find chemtools_cuda_lib
find_package(cugbasis_lib)
set(TEST_SOURCES
		test_mol_orbitals.cu
		test_gradient.cu
		test_laplacian.cu
		test_electron_density.cu
       test_hessian.cu
	test_densbased.cu
	test_kinetic_density.cu
        test_electrostatic_potential.cu
        )
# Add test executable
add_executable (tests main_test.cu catch.hpp ${TEST_SOURCES})
target_link_libraries(tests PRIVATE pybind11::embed)
target_link_libraries(tests PRIVATE cugbasis_lib)
