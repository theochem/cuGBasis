.. _usr_installation:

Installation
############


Dependencies
============

The following dependencies will be necessary for cuGBasis to build properly,

* CMake>=3.24: (https://cmake.org/)
* CUDA/DRIVERS/NVCC/CUDA-TOOLKIT: (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* Python>=3.9: (http://www.python.org/)


Quick-Installation
===================
`CuGBasis ` can be easily installed via pip:

.. code-block:: bash

    pip install qc-cugbasis

Detailed Installation
======================
The following Python dependencies are necessary:

* NumPy >= 1.16.0: http://www.numpy.org/
* Pybind11 on python: (https://github.com/pybind/pybind11)
* IOData on python: (https://github.com/theochem/iodata)

These python dependencies can be installed via:

.. code-block:: bash

    pip install numpy pybind11 qc-iodata

For testing the following are required to be installed on the Python system:

* PyTest >= 5.4.3: (https://docs.pytest.org/ <https://docs.pytest.org/)
* GBasis: (https://github.com/theochem/gbasis)
* Horton: (https://github.com/theochem/horton)
* ChemTools: (https://github.com/theochem/chemtools)

Horton is recommended to be installed using Mamba/Conda. These can be installed via:

.. code-block:: bash

    mamba install -c theochem horton
    pip install pytest
    pip install git+https://github.com/theochem/gbasis.git
    pip install git+https://github.com/theochem/chemtools.git


The following Eigen package will automatically be linked if CMake could not find the package already.

* Eigen>=3: (https://eigen.tuxfamily.org/index.php?title=Main_Page)


Downloading Code
================

The user must obtain the latest code from theochem (https://github.com/theochem/cugbasis) in Github,
As well, as obtain the dependencies from running the `git submodule` command.

.. code-block:: bash

   git clone https://github.com/theochem/CuGBasis.git

   # Get the dependencies in ./libs/ folder
   git submodule update --init --recursive



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

Compute Canada
---------------

The following is the set of instructions for creating a Python environment inside Compute Canada
and installing cuGBasis. It's important to compile/install cuGBasis with a GPU enabled.
It is recommended that CMake version be greater than 3.24 (see below).
Note that different Cuda environments can be loaded, but here we will load Cuda 11.7 version.
It's important to load the required dependencies before creating the python environment
so that the same compiler is used when creating the Python virtual environment, and when
installing (this may not be required but is hypothesized to may cause future errors).

.. code-block:: bash

    # Load the dependencies for cuGBasis and Python environment
    module load StdEnv/2020 intel/2020.1.217 cmake cuda/11.7 eigen/3.4.0
    module load python/3.9

    # Create Python environment
    virtualenv --no-download py39_cugbasis

    # Activate Environment
    source ./py39_cugbasis/bin/activate

    # Install dependencies
    pip install --no-index --upgrade pip
    pip install numpy scipy pybind11 --no-index
    pip install qc-iodata

    # Get the package and dependencies
    git clone https://github.com/theochem/cuGBasis.git
    cd cuGBasis
    git submodule update --init --recursive

    # Enable GPU
    salloc --time=1:0:0 --account=ACCOUNT --mem=12G --gres=gpu:p100:1

    # Load the required dependencies
    module load StdEnv/2020 intel/2020.1.217 cmake cuda/11.7 eigen/3.4.0
    source ~/py39_cugbasis/bin/activate

    # Go to cuGBasis folder and install it via pip
    pip install -v .


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

- Eigen is added in the lib folder and CMake will first initially try to find if Eigen was installed.
  If Eigen is not found, then it will try to link it by itself.
  If these still don't work, then add the following flag to the path containing the Eigen3*.cmake files. See
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
