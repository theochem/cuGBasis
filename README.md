
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub contributors](https://img.shields.io/github/contributors/qtchem/chemtoolscuda.svg)](https://github.com/qtchem/chemtoolscuda/graphs/contributors)

## About
ChemToolsCUDA is a free, and open-source C++/CUDA and Python library for computing various quantities efficiently 
using NVIDIA GPU's in quantum chemistry. It is highly-optimized and vectorized, making it useful for cases
where efficiency matters. ChemToolsCUDA utilizes meta-programming (code generation and template) and
GPU techniques, making it highly optimized and vectorized for cases where efficiency matters.

It depends on reading Gaussian format check-point files (.fchk) using IOData and supports up-to g-type orbitals. 
The user can use [IOData](https://www.github.com/theochem/iodata) to convert various file-types to fchk format.

To report any issues or ask questions, either [open an issue](
https://github.com/qtchem/chemtoolscuda/issues/new) or email [qcdevs@gmail.com]().

### Features
ChemToolsCUDA can compute the following quantities over any size of grid-points:

- Molecular orbitals
- Electron density
- Gradient of electron density
- Laplacian of electron density
- Electrostatic potential
- Compute density-based descriptors:
  - Reduced density gradient
  - Shannon information density
  - Norm of gradient
- Compute various kinetic energy densities:
  - Positive definite kinetic energy density
  - General kinetic energy density
  - Von Weizsacker kinetic Energy Density
  - Thomas-Fermi kinetic energy density.
  - General gradient expansion approximation of kinetic energy density

## Requirements

- CMake>=3.24: (https://cmake.org/) 
- Eigen>=3: (https://eigen.tuxfamily.org/index.php?title=Main_Page)
- CUDA/NVCC/CUDA-TOOLKIT: (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Python>=3.7: (http://www.python.org/)
- Pybind11 on python: (https://github.com/pybind/pybind11)
- IOData on python: (https://github.com/theochem/iodata)

For testing the following are required to be installed on the Python system:
- GBasis: (https://github.com/theochem/gbasis)
- Chemtools: (https://github.com/theochem/chemtools)

## Installation

```bash
git clone https://github.com/qtchem/chemtoolscuda

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

### Installation problems

The following can help with compiling this package

1. If CUBLAS, CURAND are not found, add the following flag to the correct path. 
See [here](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) for more information on how to modify CMake.
```bash 
# If pip:
CUDATOOLkit_ROOT=/some/path pip install -v .
# If cmake:
cmake -S . -B ./out/build/ -DCUDAToolkit_ROOT=/some/path 
```
2. If NVCC compiler is not found, add the following flag to correct path
```bash
# If pip:
CUDACXX=/some/path/bin/nvcc pip install -v .
# If cmake:
cmake -S . -B ./out/build/ -DCUDACXX=/some/path/bin/nvcc
```
3. If Eigen is not found, add the following flag to the path containing the Eigen3*.cmake files. See
[here](https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html) for more information.
```bash
# if pip:
CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/opt/eigen/3.3" pip install -v .
# if cmake:
cmake -S . -B ./out/build/ -DEigen3_DIR=/some/path/share/eigen3/cmake/
```
4. If you need to set the correct GPU architecture (e.g. compute capability 6.0), then add the following
Not setting the correct GPU architecture will result in an error using the constant memory. Note that if cmake is
greater than 3.24, then if it isn't provided by the user, then CMake will automatically find the correct CUDA architecture.
```bash
# if pip:
CMAKE_CUDA_ARCHITECTURES=60 pip install -v .
# if cmake:
cmake -S . -B ./out/build/ -DCMAKE_CUDA_ARCHITECTURES=60
```

## How To Use
```python
import chemtools_cuda

mol = chemtools_cuda.Molecule( FCHK FILE PATH HERE)
mol.basis_set_to_constant_memory(False)

density =  mol.compute_electron_density_on_cubic_grid( CUBIC INFO HERE)
density = mol.compute_electron_density( POINTS )
gradient = mol.compute_electron_density_gradient( POINTS )
laplacian = mol.compute_laplacian_electron_density( POINTS )
kinetic_dens = mol.compute_positive_definite_kinetic_energy_density( POINTS )
general_kin = mol.compute_general_kinetic_energy_density(POINTS, alpha)
shannon_info = mol.compute_shannon_information_density(POINTS)
reduced_dens = mol.compute_reduced_density_gradient(POINTS)

# Parameter True needs to be set to use ESP, this will hold true for any integrals
mol.basis_set_to_constant_memory(True)
esp = mol.compute_electrostatic_potential( POINTS )
```
