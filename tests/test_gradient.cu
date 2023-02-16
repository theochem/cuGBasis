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
#include "../include/evaluate_gradient.cuh"

namespace py = pybind11;
using namespace py::literals;


TEST_CASE( "Test Derivative of Contractions Against gbasis", "[evaluate_contraction_derivatives]" ) {
  //py::initialize_interpreter();  // Open up the python interpretor for this test.
  {  // Need this so that the python object doesn't outline the interpretor.
    // Get the IOdata object from the fchk file.
    std::string fchk_file = GENERATE(
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_he.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be_f_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_kr.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_o.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_c_g_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_mg.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/E948_rwB97XD_def2SVP.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/h2o.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/ch4.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000092_HF_cc-pVDZ.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "FCHK FILE %s \n" << fchk_file << std::endl;
    gbasis::IOData iodata = gbasis::get_molecular_basis_from_fchk(fchk_file);

    // Gemerate random grid.
    int numb_pts = 10000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);

    // Calculate Gradient
    gbasis::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
    std::vector<double> gradient_result = gbasis::evaluate_contraction_derivatives(iodata, points.data(), numb_pts);

    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = gbasis::as_pyarray_from_vector(gradient_result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = gbasis::as_pyarray_from_vector(points);

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

#np.set_printoptions(threshold=np.inf)
#print(true_result)
#print(true_result.shape)

#print(true_result[:10])
true_result = true_result.reshape((3, nbasis, numb_pts), order="C")  # column-major order
iodata = load_one(fchk_path)
basis, type = from_iodata(iodata)

points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

# Derivative in X-coordinate
derivative =  evaluate_deriv_basis(
            basis, points, np.array([1, 0, 0]), coord_type=type
        )
error = np.abs(derivative - true_result[0, :, :])
#print( true_result[0, :, :])
#print(derivative)
#print(error)
#counter = 0
#for i in range(0, len(basis)):
#  shell = basis[i]
#  types = type[i]
#  angmom = shell.angmom
#  print("Angmom ", angmom)
  #print(shell.__dict__)

#  for k in range(0, shell.num_sph):
#    #print(true_result[0, counter, :], derivative[counter, :])
#    print(counter, true_result[0, counter, :], derivative[counter, :], np.abs(true_result[0, counter, :] - derivative[counter, :]))
#    counter += 1
#  print("")
print(np.max(error), np.mean(error), np.std(error))
assert np.all(error < 1e-10), "Gradient on electron density on GPU doesn't match gbasis."


# # Test using finite-difference.
#density = evaluate_basis(basis, points, coord_type=type)
#for i in [0, 1, 2]:
#  step = np.array([0., 0., 0.])
#  step[i] = 1e-8
#  density_2 = evaluate_basis(basis, points + step, coord_type=type)
#  finite_diff = (density_2 - density) / 1e-8
#  err = np.abs(finite_diff - true_result[i, :, :])
#  print(np.max(err), np.mean(err), np.min(err))
#  assert np.all(err < 1e-4), f"Gradient on electron density on GPU doesn't match finite difference in {i}th index."
    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
  //py::finalize_interpreter(); // Close up the python interpretor for this test.
}



TEST_CASE( "Test Gradient of Electron Density Against gbasis", "[evaluate_electron_density_gradient]" ) {
  //py::initialize_interpreter();  // Open up the python interpretor for this test.
  {  // Need this so that the python object doesn't outline the interpretor.
    // Get the IOdata object from the fchk file.
    std::string fchk_file = GENERATE(
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_he.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be_f_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_kr.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_o.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_c_g_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_mg.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/E948_rwB97XD_def2SVP.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/h2o.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/ch4.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000092_HF_cc-pVDZ.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
    std::cout << "FCHK FILE %s \n" << fchk_file << std::endl;
    gbasis::IOData iodata = gbasis::get_molecular_basis_from_fchk(fchk_file);

    // Gemerate random grid.
    int numb_pts = 1000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);

    // Calculate Gradient
    printf("Calculate Gradient \n");
    gbasis::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
    std::vector<double> gradient_result = gbasis::evaluate_electron_density_gradient(iodata, points.data(), numb_pts);
    printf("Done Gradient \n");

    // COnvert them (with copy) to python objects so that they can be transfered.
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_result = gbasis::as_pyarray_from_vector(gradient_result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = gbasis::as_pyarray_from_vector(points);

    auto locals = py::dict("points"_a = py_points,
                           "true_result"_a = py_result,
                           "fchk_path"_a = fchk_file,
                           "numb_pts"_a = numb_pts);
    py::exec(R"(
import numpy as np
from gbasis.evals.density import evaluate_density_gradient, evaluate_density
from iodata import load_one
from gbasis.wrappers import from_iodata

true_result = true_result.reshape((numb_pts, 3), order="C")  # Row-major order

iodata = load_one(fchk_path)
basis, type = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

gradient = evaluate_density_gradient(rdm, basis, points, coord_type=type)
error = np.abs(gradient - true_result)
print("Max, Mean, STD , Min error ", np.max(error), np.mean(error), np.std(error), np.min(error))
assert np.all(error < 1e-10), "Gradient on electron density on GPU doesn't match gbasis."


# # Test using finite-difference.
#density = evaluate_density(rdm, basis, points, coord_type=type)
#for i in [0, 1, 2]:
#  step = np.array([0., 0., 0.])
#  step[i] = 1e-8
#  density_2 = evaluate_density(rdm, basis, points + step, coord_type=type)
#  finite_diff = (density_2 - density) / 1e-8
#  err = np.abs(finite_diff - true_result[:, i])
#  print(np.max(err), np.mean(err), np.min(err))
#  assert np.all(np.abs(finite_diff - true_result[:, i]) < 1e-4), f"Gradient on electron density on GPU doesn't match finite difference in {i}th index."
    )", py::globals(), locals);
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
  //py::finalize_interpreter(); // Close up the python interpretor for this test.
}
