
# Requirements
- Eigen needs to be installed and be on your path. TODO: FIgur eout how to build and instlal 
Eigen using Cmake automatically.
- NVCC compiler with single NVIDIA GPU. TODO: only works with 12GB or Higher GPU because I fixed the memory.
- Pybind11 
- IOData
- GBasis and Chemtools for tests

# Installation
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
