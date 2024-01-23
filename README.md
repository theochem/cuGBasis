
# Requirements
- [CMake>=3.24](https://cmake.org/)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 
- [CUDA/NVCC/CUDA-TOOLKIT](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
- [Pybind11](https://github.com/pybind/pybind11) 
- [IOData](https://github.com/theochem/iodata)

For testing the following are required:
- [GBasis](https://github.com/theochem/gbasis)
- [Chemtools](https://github.com/theochem/chemtools)

# Installation

```bash
git clone https://github.com/qtchem/gbasis_cuda

# Get the dependencies in ./libs/ folder
git submodule update --init --recursive
```

In-order to install it need to use:
```bash
pip install -v . 
```
The -v is needed in order to debug by looking at the output of CMake.
If CMake can't find NVCC or C++ compiler, then `CMakeLists.txt` needs to be modified
to find them.

In order to test do
```bash
pytest ./tests/*.py -v 
```

In order to build without the python bindings, useful for debugging purposes,
```bash
cmake -S . -B ./out/build/  
make -C ./out/build/
./out/build/tests/tests  # Run the C++/CUDA tests
```

### Building with CMake

The following can help with compiling this package

1. If CUBLAS, CURAND are not found, add the following flag to the correct path
```bash 
cmake -S . -B ./out/build/ -DCUDAToolkit_ROOT=/some/path 
```
2. If NVCC is not found, add the following flag to correct path
```bash
cmake -S . -B ./out/build/ -DCUDACXX=/some/path/bin/nvcc
```
3. If Eigen is not found, add the following flag to the path containing the Eigen3*.cmake files
```bash
cmake -S . -B ./out/build/ -DEigen3_DIR=/some/path/share/eigen3/cmake/
```
4. 

# How To Use
```python
import gbasis_cuda

mol = gbasis_cuda.Molecule( FCHK FILE PATH HERE)
mol.basis_set_to_constant_memory(False)

density =  mol.compute_electron_density_on_cubic_grid( CUBIC INFO HERE)
density = mol.compute_electron_density( POINTS )
gradient = mol.compute_electron_density_gradient( POINTS )
laplacian = mol.compute_laplacian_electron_density( POINTS )
kinetic_dens = mol.compute_positive_definite_kinetic_energy_density( POINTS )
general_kin = mol.compute_general_kinetic_energy_density(POINTS, alpha)

# Parameter True needs to be set to use ESP, this will hold true for any integrals
mol.basis_set_to_constant_memory(True)
esp = mol.compute_electrostatic_potential( POINTS )
```
