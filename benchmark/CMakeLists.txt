
# Specify the benchmark source file
find_package(cugbasis_lib)

# Add test executable
add_executable (benchmark benchmark_main.cu )
# If the user didn't define CUDA architecture
if("$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    # If cmake version is greater than 3.24, automatically find the right CUDA Architecture
    if("$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES native)
        endif()
    else()
        set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES 52 60 61 75)  # Set a default CUDA_ARCHITECTURE
    endif ()
else()
    message("CUDA Architecture specified: $ENV{CMAKE_CUDA_ARCHITECTURES}")
    # If the user did define the CUDA Archiecture then add it here.
    set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
endif()
target_link_libraries(benchmark PUBLIC cugbasis_lib)
#target_link_libraries(benchmark PUBLIC ./out/build/libcugbasis_lib.so)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
         -v -rdc=true --expt-relaxed-constexpr -lineinfo #-opt-info
        >)
target_link_libraries(benchmark PUBLIC pybind11::embed)
