#include "catch.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "../include/iodata.h"
#include "../include/evaluate_density.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/basis_to_gpu.cuh"
#include "../include/utils.h"

namespace py = pybind11;
using namespace py::literals;


TEST_CASE( "Test Electron Density Against gbasis", "[evaluate_electron_density_on_cubic]" ) {
  //py::initialize_interpreter();  // Open up the python interpretor for this test.
  {  // Need this so that the python object doesn't outline the interpretor.
  // Evaluate the electron density of this example.
  std::string fchk_file = GENERATE(
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_he.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_be_f_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_kr.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_o.fchk",
//        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_c_g_pure_orbital.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_mg.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/E948_rwB97XD_def2SVP.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/h2o.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/ch4.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000092_HF_cc-pVDZ.fchk"
        //"/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
    );
  printf("IODATA OB %s \n", fchk_file.c_str());
  gbasis::IOData iodata = gbasis::get_molecular_basis_from_fchk(fchk_file);
  gbasis::UniformGrid grid =
      gbasis::get_grid_from_coordinates_charges(iodata.GetCoordAtoms(), iodata.GetCharges(), iodata.GetNatoms(), 0.0 );
  printf("DOne grid \n");
  double3 l_bnd = {grid.l_bnd[0], grid.l_bnd[1], grid.l_bnd[2]};
  std::array<int, 3> shape_arr = grid.calculate_shape({0.9, 0.9, 0.9});
  std::array<double, 9> axes_spacing = grid.multiply_axes_by_spacing(0.9, 0.9, 0.9);
  int3 shape = {shape_arr[0], shape_arr[1], shape_arr[2]};

  // Evaluate electron density on the cube
  printf("Electron density \n");
  gbasis::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, false);
  std::vector<double> result = gbasis::evaluate_electron_density_on_cubic(iodata, l_bnd, &axes_spacing[0], shape, false);
  printf("DOne \n");

  //Transfer result to pybind11 without copying
  pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> py_result = gbasis::as_pyarray_from_vector(result);
  py::array_t<double> l_bnd_py(3, &l_bnd.x);
  py::array_t<double> axes_with_spacing_py(axes_spacing.size(), axes_spacing.data());
  py::array_t<int> shape_py(3, &shape.x);

  auto locals = py::dict(
      "true_result"_a=py_result, "fchk_path"_a=fchk_file, "l_bnd"_a=l_bnd_py, "axes"_a=axes_with_spacing_py, "shape"_a=shape_py
  );
  py::exec(R"(
import numpy as np
from gbasis.evals.density import evaluate_density
from iodata import load_one
from gbasis.wrappers import from_iodata

iodata = load_one(fchk_path)
basis, type = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)

num_pts = 50
axes = np.reshape(axes, (3, 3)).T

random_indices = np.array([np.random.randint(0, shape[0], num_pts), np.random.randint(0, shape[1], num_pts), np.random.randint(0, shape[2], num_pts)]).T
global_index = random_indices[:, 0] * (shape[1] * shape[2]) + shape[2] * random_indices[:, 1] + random_indices[:, 2]
grid = l_bnd + random_indices.dot(axes)

density = evaluate_density(rdm, basis, grid, coord_type=type)

result = np.all(np.abs(density - true_result[global_index]) < 1e-8)
print(np.abs(density - true_result[global_index]))
assert result, "Electron density on GPU doesn't match gbasis."
    )", py::globals(), locals);

  if (!locals["result"].cast<bool>()) {
    REQUIRE(true);
  }
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
  //py::finalize_interpreter(); // Close up the python interpretor for this test.
}



TEST_CASE( "Test Electron Density Against gbasis on random grid", "[evaluate_electron_density_on_any_grid]" ) {
  //py::initialize_interpreter();  // Open up the python interpretor for this test.
  {  // Need this so that the python object doesn't outline the interpretor.
    // Evaluate the electron density of this example.
    std::string fchk_file = GENERATE(
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
        "/home/ali-tehrani/SoftwareProjects/spec_database/tests/data/atom_he.fchk"
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
    printf("IODATA OB %s \n", fchk_file.c_str());
    gbasis::IOData iodata = gbasis::get_molecular_basis_from_fchk(fchk_file);

    // Gemerate random grid.
    int numb_pts = 10000;
    std::vector<double> points(3 * numb_pts);
    std::random_device rnd_device;
    std::mt19937  merseene_engine {rnd_device()};
    std::uniform_real_distribution<double> dist {-5, 5};
    auto gen = [&dist, &merseene_engine](){return dist(merseene_engine);};
    std::generate(points.begin(), points.end(), gen);


    // Evaluate electron density on the cube
    printf("Electron density \n");
    gbasis::add_mol_basis_to_constant_memory_array(iodata.GetOrbitalBasis(), false, true);
    std::vector<double> result = gbasis::evaluate_electron_density_on_any_grid(iodata, points.data(), numb_pts);
    printf("DOne \n");

    //Transfer result to pybind11 without copying
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> py_result =
        gbasis::as_pyarray_from_vector(result);
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        py_points = gbasis::as_pyarray_from_vector(points);

    auto locals = py::dict(
        "true_result"_a=py_result, "fchk_path"_a=fchk_file, "points"_a = py_points, "numb_pts"_a = numb_pts
    );
    py::exec(R"(
import numpy as np
from gbasis.evals.density import evaluate_density
from iodata import load_one
from gbasis.wrappers import from_iodata

iodata = load_one(fchk_path)
basis, type = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
points = points.reshape((numb_pts, 3), order="F")
points = np.array(points, dtype=np.float64)

density = evaluate_density(rdm, basis, points, coord_type=type)

result = np.all(np.abs(density - true_result) < 1e-8)
print(np.abs(density - true_result))
assert result, "Electron density on GPU doesn't match gbasis."
    )", py::globals(), locals);

    if (!locals["result"].cast<bool>()) {
      REQUIRE(true);
    }
  } // Need this so that the python object doesn't outline the interpretor when we close it up.
  //py::finalize_interpreter(); // Close up the python interpretor for this test.
}
