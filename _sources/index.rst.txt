.. CuGBasis documentation master file, created by
   sphinx-quickstart on Mon Mar 25 15:07:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to CuGBasis's documentation!
=========================================

.. image:: cuGBasis_Logo.jpeg
    :width: 450

CuGBasis is a free, and open-source C++/CUDA and Python library for computing various quantities efficiently
using NVIDIA GPU's in quantum chemistry. It is highly-optimized and vectorized, making it useful for cases
where efficiency matters. It has substantial speed-ups compared to both commerical and open-source post-processing
quantum chemistry codes.

CuGBasis can read various wave-function formats (wfn, wfx, molden and fchk) using IOData and supports up-to g-type
(Cartesian or Pure) orbitals. It can compute the following features:

* Molecular orbitals
* Electron density
* Gradient of electron density
* Laplacian of electron density
* Hessian of electron density
* Electrostatic potential
* Compute density-based descriptors:

  * Reduced density gradient
  * Shannon information density
  * Norm of gradient

* Compute various kinetic energy densities:

  * Positive definite kinetic energy density
  * General kinetic energy density
  * Von Weizsacker kinetic Energy Density
  * Thomas-Fermi kinetic energy density.
  * General gradient expansion approximation of kinetic energy density


To report any issues or ask questions, either `open an issue <https://github.com/qtchem/cugbasis/issues/new>`_
or email qcdevs@gmail.com.

.. toctree::
   :maxdepth: 2
   :caption: User documentation:

   ./installation.rst
   ./molecule.rst
   ./conventions.rst

.. toctree::
    :maxdepth: 2
    :caption: Example Tutorials:

    ./quick_start
    ./critical_points
    ./reduced_density_gradient
    ./generating_cube_files


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
