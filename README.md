<img src="./doc/cuGBasis_Logo.jpeg" alt="Image" width="300"/>

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.9.html)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub contributors](https://img.shields.io/github/contributors/theochem/cugbasis.svg)](https://github.com/theochem/cugbasis/graphs/contributors)

## About
CuGBasis is a free, and open-source C++/CUDA and Python library for computing efficient computation of scalar, vector, and matrix quantities
using NVIDIA GPU's in quantum chemistry. It is highly-optimized and vectorized, making it useful for cases
where efficiency matters. 

CuGBasis can compute the molecular orbitals, electron density (and its derivatives), electrostatic
potentials and many other types of quantum chemistry descriptors and  can read various wave-function formats (wfn, wfx, molden and fchk) using 
IOData and supports up-to g-type orbitals. 

See the website for more information: [cuGBasis](https://cugbasis.qcdevs.org)

To report any issues or ask questions, either [open an issue](
https://github.com/theochem/cuGBasis/issues/new) or email [qcdevs@gmail.com]().


## Citation
Please use the following citation in any publication:

```
 @article{cugbasis,
    author = {Tehrani, Alireza and Richer, Michelle and Heidar-Zadeh, Farnaz},
    title = "{CuGBasis: High-performance CUDA/Python library for efficient computation of quantum chemistry density-based descriptors for larger systems}",
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {7},
    pages = {072501},
    year = {2024},
    month = {08},
    issn = {0021-9606},
    doi = {10.1063/5.0216781},
    url = {https://doi.org/10.1063/5.0216781},
}
```

