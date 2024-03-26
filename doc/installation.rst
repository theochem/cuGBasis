.. _usr_installation:

Installation
############

Downloading Code
================

The latest code can be obtained through theochem (https://github.com/qtchem/ChemtoolsCUDA) in Github,

.. code-block:: bash

   git clone https://github.com/qtchem/ChemToolsCUDA.git

   # Get the dependencies in ./libs/ folder
   git submodule update --init --recursive



Dependencies
============

The following dependencies will be necessary for ChemToolsCUDA to build properly,

* CMake>=3.24: (https://cmake.org/)
* Eigen>=3: (https://eigen.tuxfamily.org/index.php?title=Main_Page)
* CUDA/DRIVERS/NVCC/CUDA-TOOLKIT: (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* Python>=3.7: (http://www.python.org/)
* NumPy >= 1.16.0: http://www.numpy.org/
* Pybind11 on python: (https://github.com/pybind/pybind11)
* IOData on python: (https://github.com/theochem/iodata)

For testing the following are required to be installed on the Python system:

* PyTest >= 5.4.3: (https://docs.pytest.org/ <https://docs.pytest.org/)
* GBasis: (https://github.com/theochem/gbasis)
* ChemTools: (https://github.com/theochem/chemtools)


Installation
============

Python Binding
---------------

Once you downloaded the code, obtained the required libraries and dependencies, then one can use pip to install
the Python bindings:

.. code-block:: bash

    pip install -v .


In order to see if it successfully installed, run the following

.. code-block:: bash

    pytest -v ./tests/*.py


C++
---

In order to build a shared library without the python bindings, particularly useful for debugging purposes,

.. code-block:: bash

    cmake -S . -B ./out/build/
    make -C ./out/build/
    ./out/build/tests/tests  # Run the C++/CUDA tests



Installation problems
=====================

The following can help with compiling this package

- If CMake version is greater than 3.24, then CMake will automatically find the correct CUDA architecture based on the
  user's NVIDIA GPU.
  If not, the user will need to set the correct GPU architecture (e.g. compute capability 6.0). This can be
  found through the `NVIDIA website <https://developer.nvidia.com/cuda-gpus>`_. Once it is found, then one can
  add an environment variable to indicate to compile using the correct CUDA architecture.

.. code-block:: bash

    # if pip:
    CMAKE_CUDA_ARCHITECTURES=60 pip install -v .
    # if cmake:
    cmake -S . -B ./out/build/ -DCMAKE_CUDA_ARCHITECTURES=60

- If CUBLAS, CURAND are not found, add the following flag to the correct path.
  See `here <https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html>`_ for more information on how to modify CMake.

.. code-block:: bash

    # If pip:
    CUDATOOLkit_ROOT=/some/path pip install -v .
    # If cmake:
    cmake -S . -B ./out/build/ -DCUDAToolkit_ROOT=/some/path

- If NVCC compiler is not found, add the following flag to correct path

.. code-block:: bash

    # If pip:
    CUDACXX=/some/path/bin/nvcc pip install -v .
    # If cmake:
    cmake -S . -B ./out/build/ -DCUDACXX=/some/path/bin/nvcc

- If Eigen is not found, add the following flag to the path containing the Eigen3*.cmake files. See
  `here <https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html>`_ for more information.

.. code-block:: bash

    # if pip:
    CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/opt/eigen/3.3" pip install -v .
    # if cmake:
    cmake -S . -B ./out/build/ -DEigen3_DIR=/some/path/share/eigen3/cmake/



Building Documentation
======================

The following shows how to build the documentation using sphinx to the folder `_build`.

    .. code-block:: bash

        cd doc
        make html
