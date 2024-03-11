#include "catch.hpp"

#include <random>
#include <algorithm>
#include <iterator>

#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "../include/iodata.h"
#include "../include/utils.h"
#include "../include/cuda_utils.cuh"
#include "../include/basis_to_gpu.cuh"
#include "../include/evaluate_electrostatic.cuh"

namespace py = pybind11;
using namespace py::literals;


TEST_CASE( "Test Electrostatic Potential", "[evaluate_electrostatic_potential]" ) {
  //py::initialize_interpreter();  // Open up the python interpretor for this test.
  {  // Need this so that the python object doesn't outline the interpretor.
    // Get the IOdata object from the fchk file.
    std::string fchk_file = GENERATE(
        "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/atom_he.fchk",
        "./tests/data/atom_be.fchk",
        "./tests/data/atom_be_f_pure_orbital.fchk",
        "./tests/data/atom_be_f_cartesian_orbital.fchk",
        "./tests/data/atom_kr.fchk",
        "./tests/data/atom_o.fchk",
        // TODO: Support for g-type
        // "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_c_g_pure_orbital.fchk",
        "./tests/data/atom_mg.fchk",
        "./tests/data/E948_rwB97XD_def2SVP.fchk",
        "./tests/data/test.fchk",
        "./tests/data/test2.fchk",
        "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/h2o.fchk",
        "./tests/data/ch4.fchk",
        "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
        // TODO: Support for g-type
        //"/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "FCHK FILE %s \n" << fchk_file << std::endl;
    chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk(fchk_file);

    // Gemerate random grid.
    int numb_pts = 1000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);

    // Calculate Gradient
    std::vector<double> esp_result = chemtools::compute_electrostatic_potential_over_points(
        iodata, points.data(), numb_pts
        );

    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = chemtools::as_pyarray_from_vector(esp_result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = chemtools::as_pyarray_from_vector(points);

    const int nbasis = iodata.GetOrbitalBasis().numb_basis_functions();

    auto locals = py::dict("points"_a = py_points,
                           "true_result"_a = py_result,
                           "fchk_path"_a = fchk_file,
                           "numb_pts"_a = numb_pts,
                           "nbasis"_a = nbasis);
    py::exec(R"(
import numpy as np
from gbasis.evals.electrostatic_potential import electrostatic_potential
from iodata import load_one
from gbasis.wrappers import from_iodata

iodata = load_one(fchk_path)
basis = from_iodata(iodata)
points = points.reshape((numb_pts, 3), order="C")
points = np.array(points, dtype=np.float64)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)

#from chemtools.wrappers import Molecule
#mol2 = Molecule.from_file(fchk_path)
#electro = mol2.compute_esp(points)
electro = electrostatic_potential(basis=basis, one_density_matrix=rdm, points=points, nuclear_coords=iodata.atcoords,
                                  nuclear_charges=iodata.atcorenums)
err = np.abs(electro - true_result)
print("Max error: ", np.max(err))
result = np.all(err < 1e-8)
assert result, "Electrostatic potential on GPU doesn't match gbasis."
    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
  //py::finalize_interpreter(); // Close up the python interpretor for this test.
}

