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
#include "../include/evaluate_hessian.cuh"

namespace py = pybind11;
using namespace py::literals;


TEST_CASE( "Test Second Derivative of Contractions Against gbasis", "[evaluate_contraction_second_derivative]" ) {
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
        "./tests/data/atom_c_g_pure_orbital.fchk",
        "./tests/data/atom_mg.fchk",
        "./tests/data/E948_rwB97XD_def2SVP.fchk",
        "./tests/data/test.fchk",
        "./tests/data/test2.fchk",
        "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/h2o.fchk",
        "./tests/data/ch4.fchk",
        "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
        "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "Second Derivative Contractions FCHK file %s \n" << fchk_file << std::endl;
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
    chemtools::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
    std::vector<double> sec_deriv = chemtools::evaluate_contraction_second_derivative(iodata, points.data(), numb_pts);

    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = chemtools::as_pyarray_from_vector(sec_deriv);
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
from gbasis.evals.density import evaluate_deriv_basis, evaluate_basis
from iodata import load_one
from gbasis.wrappers import from_iodata

true_result = true_result.reshape((6, nbasis, numb_pts), order="C")  # row-major order
iodata = load_one(fchk_path)
basis = from_iodata(iodata)

points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

for i, deriv in enumerate([[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]):
  derivative =  evaluate_deriv_basis(
              basis, points, np.array(deriv)
          )
  error = np.abs(derivative - true_result[i, :, :])
  print(deriv, np.max(error), np.mean(error), np.std(error))
  assert np.all(error < 1e-10), "Second Derivative of contractions on GPU doesn't match gbasis."

    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
}


TEST_CASE( "Test Hessian (Row-Order) of Electron Density Against gbasis", "[evaluate_electron_density_hessian_row]" ) {
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
        "./tests/data/atom_c_g_pure_orbital.fchk",
        "./tests/data/atom_mg.fchk",
        "./tests/data/E948_rwB97XD_def2SVP.fchk",
        "./tests/data/test.fchk",
        "./tests/data/test2.fchk",
        "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/h2o.fchk",
        "./tests/data/ch4.fchk",
        "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
        "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "Compute Hessian on (Row-Order) FCHK file %s \n" << fchk_file << std::endl;
    chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk(fchk_file);

    // Generate random grid.
    int numb_pts = 1000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);

    // Calculate Gradient
    chemtools::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
    std::vector<double> hessian_result = chemtools::evaluate_electron_density_hessian(iodata, points.data(), numb_pts, true);
    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = chemtools::as_pyarray_from_vector(hessian_result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = chemtools::as_pyarray_from_vector(points);

    auto locals = py::dict("points"_a = py_points,
                           "true_result"_a = py_result,
                           "fchk_path"_a = fchk_file,
                           "numb_pts"_a = numb_pts);
    py::exec(R"(
import numpy as np
from gbasis.evals.density import evaluate_density_hessian
from gbasis.evals.density import evaluate_deriv_basis, evaluate_basis
from iodata import load_one
from gbasis.wrappers import from_iodata

true_result = true_result.reshape((numb_pts, 3, 3), order="C")  # Row-major order

iodata = load_one(fchk_path)
basis = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

indices_to_compute = np.random.choice(np.arange(len(points)), size=10000)
true_result = true_result[indices_to_compute, :]
points = points[indices_to_compute, :]

hessian = evaluate_density_hessian(rdm, basis, points)
error = np.abs(hessian - true_result)
print("Max, Mean, STD , Min error ", np.max(error), np.mean(error), np.std(error), np.min(error))
assert np.all(error < 1e-10), "Hessian on electron density on GPU doesn't match gbasis."
    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
}


TEST_CASE( "Test Hessian (Col-Order) of Electron Density Against gbasis", "[evaluate_electron_density_hessian_row]" ) {
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
        "./tests/data/atom_c_g_pure_orbital.fchk",
        "./tests/data/atom_mg.fchk",
        "./tests/data/E948_rwB97XD_def2SVP.fchk",
        "./tests/data/test.fchk",
        "./tests/data/test2.fchk",
        "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
        "./tests/data/h2o.fchk",
        "./tests/data/ch4.fchk",
        "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
        "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "Compute Hessian on (Col-Order) FCHK file %s \n" << fchk_file << std::endl;
    chemtools::IOData iodata = chemtools::get_molecular_basis_from_fchk(fchk_file);

    // Generate random grid.
    int numb_pts = 500000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);

    // Calculate Gradient
    chemtools::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
    std::vector<double> hessian_result = chemtools::evaluate_electron_density_hessian(iodata, points.data(), numb_pts, false);
    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = chemtools::as_pyarray_from_vector(hessian_result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = chemtools::as_pyarray_from_vector(points);

    auto locals = py::dict("points"_a = py_points,
                           "true_result"_a = py_result,
                           "fchk_path"_a = fchk_file,
                           "numb_pts"_a = numb_pts);
    py::exec(R"(
import numpy as np
from gbasis.evals.density import evaluate_density_hessian
from gbasis.evals.density import evaluate_deriv_basis, evaluate_basis
from iodata import load_one
from gbasis.wrappers import from_iodata

true_result = true_result.reshape((numb_pts, 3, 3), order="F")  # Row-major order

iodata = load_one(fchk_path)
basis = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

indices_to_compute = np.random.choice(np.arange(len(points)), size=10000)
true_result = true_result[indices_to_compute, :]
points = points[indices_to_compute, :]

hessian = evaluate_density_hessian(rdm, basis, points)
error = np.abs(hessian - true_result)
print("Max, Mean, STD , Min error ", np.max(error), np.mean(error), np.std(error), np.min(error))
assert np.all(error < 1e-10), "Hessian on electron density on GPU doesn't match gbasis."
    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
}