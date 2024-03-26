
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub contributors](https://img.shields.io/github/contributors/qtchem/chemtoolscuda.svg)](https://github.com/qtchem/chemtoolscuda/graphs/contributors)

![Image](./doc/chemtoolscuda_logo_with_text.jpeg)

## About
ChemToolsCUDA is a free, and open-source C++/CUDA and Python library for computing various quantities efficiently 
using NVIDIA GPU's in quantum chemistry. It is highly-optimized and vectorized, making it useful for cases
where efficiency matters.

ChemToolsCUDA can read various wave-function formats (wfn, wfx, molden and fchk) using IOData and supports up-to g-type orbitals. 
Please see the website for more information.

To report any issues or ask questions, either [open an issue](
https://github.com/qtchem/chemtoolscuda/issues/new) or email [qcdevs@gmail.com]().

  
## Citation
Please use the following citation in any publication:

> **"ChemToolsCUDA: High performance CUDA/Python Library For Computing Quantum Chemistry Density-Based Descriptors for 
> Larger Systems Using GPUS."**,
> A. Tehrani, M. Richer, P. W. Ayers, F. Heidar‐Zadeh
> 
>

## How To Use
```python
import chemtools_cuda

mol = chemtools_cuda.Molecule( FCHK FILE PATH HERE)

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
