# Add catch as an interface library
set(CATCH_INCLUDE_DIR .)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Find chemtools_cuda_lib
find_package(cugbasis_lib)
set(TEST_SOURCES
		test_eval_mo.cu
        test_eval_rho_grad.cu
        test_eval_rho_lap.cu
		test_eval_rho.cu
        test_eval_rho_hess.cu
		test_densbased.cu
		test_eval_kin_energ.cu
		test_esp.cu
        )
# Add test executable
add_executable (tests main_test.cu catch.hpp ${TEST_SOURCES})
target_link_libraries(tests PRIVATE pybind11::embed)
target_link_libraries(tests PRIVATE cugbasis_lib)
