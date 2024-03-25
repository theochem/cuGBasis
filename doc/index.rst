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
where efficiency matters.

ChemToolsCUDA can read various wave-function formats (wfn, wfx, molden and fchk) using IOData and supports up-to g-type orbitals.

To report any issues or ask questions, either [open an issue](
https://github.com/qtchem/chemtoolscuda/issues/new) or email [qcdevs@gmail.com]().

.. toctree::
   :maxdepth: 2
   :caption: User documentation:

   ./molecule.rst
   ./installation.rst
   ./conventions.rst

.. toctree::
    :maxdepth: 2
    :caption: Example Tutorials:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
