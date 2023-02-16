GPU_ARCH ?= sm_20

boys_functions_example: boys_functions_example.cu boys_functions.h
	nvcc -I . -o boys_functions_example -arch=$(GPU_ARCH) boys_functions_example.cu

clean:
	rm -f boys_functions_example *~

