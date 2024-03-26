.. chemtoolscuda documentation master file, created by
   sphinx-quickstart on Mon Mar 25 15:07:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Welcome to ChemToolsCuda's documentation!
=========================================

.. image:: chemtoolscuda_logo_with_text.jpeg
    :width: 450

ChemToolsCUDA is a free, and open-source C++/CUDA and Python library for computing various quantities efficiently
using NVIDIA GPU's in quantum chemistry. It is highly-optimized and vectorized, making it useful for cases
where efficiency matters. It has substantial speed-ups compared to both commerical and open-source post-processing
quantum chemistry codes.

ChemToolsCUDA can read various wave-function formats (wfn, wfx, molden and fchk) using IOData and supports up-to g-type
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


To report any issues or ask questions, either `open an issue <https://github.com/qtchem/chemtoolscuda/issues/new>`_
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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
