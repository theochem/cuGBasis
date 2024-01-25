#include "../include/utils.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

/// Go over each atom in each direction (x, y, z) and find maximum and minimum and add by the amount.
std::pair<chemtools::CoordsXYZ, chemtools::CoordsXYZ> chemtools::get_lower_and_upper_bounds(
    const double* coordinates, const int* charges, int natoms, double add_amount) {
  chemtools::CoordsXYZ lower_bnd = {0., 0., 0.};
  chemtools::CoordsXYZ upper_bnd = {0., 0., 0.};

  // Construct new bounds based on the charges.
  std::vector<std::array<double, 3>> l_bounds_all_atoms;
  std::vector<std::array<double, 3>> u_bounds_all_atoms;
  for(int i = 0; i < natoms; i++){
    std::array<double, 3> l_atom = {coordinates[3 * i],
                                    coordinates[3 * i + 1],
                                    coordinates[3 * i + 2]};
    std::array<double, 3> u_atom =  {coordinates[3 * i],
                                     coordinates[3 * i + 1],
                                     coordinates[3 * i + 2]};
//    printf("CHARGE %d\n", charges[i]);
    double charge_diff;
    if (charges[i] == 1){
      charge_diff = 2.6;
    }
    else if (charges[i] == 5) {
      // Boron
      charge_diff = 2.8;
    }
    else if (charges[i] == 6) {
      charge_diff = 2.8;
    }
    else if (charges[i] == 7) {
      charge_diff = 3.0;
    }
    else if (charges[i] == 8) {
      // Oxygen
      charge_diff = 3.3;
    }
    else if (charges[i] == 9) {
      charge_diff = 2.9;
    }
    else if (charges[i] == 14) {
      charge_diff = 3.3;
    }
    else if (charges[i] == 15) {
      // Phosphorous
      charge_diff = 3.3;
    }
    else if (charges[i] == 16) {
      // Sulfer
      charge_diff = 3.4;
    }
    else if (charges[i] == 17) {
      // Cloride
      charge_diff = 3.5;
    }
    else if (charges[i] == 34) {
      // Selenium
      charge_diff = 3.5;
    }
    else if (charges[i] == 33) {
      // Arsenic
      charge_diff = 3.3;
    }
    else if (charges[i] == 35) {
      // Bromine
      charge_diff = 3.5;
    }
    else if (charges[i] == 51) {
      charge_diff = 3.7;
    }
    else if (charges[i] == 52) {
      // Tellerium
      charge_diff = 3.8;
    }
    else if (charges[i] == 53) {
      //IOdine
      charge_diff = 3.8;
    }
    else {
      throw std::runtime_error("Can't recognize this atom with charge " + std::to_string(charges[i]));
    }
//    printf("Atom with Charge %d with charge_Diff of %f\n", charges[i], charge_diff);
    // Go through the atom bounds and substract the lower bound and add to teh lower bound.
    for(int j = 0; j < 3; j++) {
      l_atom[j] -= charge_diff;
      u_atom[j] += charge_diff;
    }
//    printf("Lower bound (%f, %f, %f), of Atom with CHarge %d\n", l_atom[0], l_atom[1], l_atom[2]);
//    printf("Upper bound (%f, %f, %f), of Atom with CHarge %d\n", u_atom[0], u_atom[1], u_atom[2]);

    // Store the bounding box over each atom.
    l_bounds_all_atoms.push_back(l_atom);
    u_bounds_all_atoms.push_back(u_atom);

  }


  // Go through each bounding box over each atom adn grab the one that is the smallest in each dimension.
  for(int j = 0; j < natoms; j+=1) {
    for(int i = 0; i < 3; i++) {
      if (l_bounds_all_atoms[j][i] < lower_bnd[i]) {
        lower_bnd[i] = l_bounds_all_atoms[j][i];
      }
      if (u_bounds_all_atoms[j][i] > upper_bnd[i]) {
        upper_bnd[i] = u_bounds_all_atoms[j][i];
      }
    }
  }

  // If optional argument add_amount is non-zero, then make the lower bound lower and upper bound even "upper"
  if (add_amount > 0) {
    for(int i = 0; i < 3; i++) {
      lower_bnd[i] -= add_amount;
      upper_bnd[i] += add_amount;
    }
  }
  return std::pair<chemtools::CoordsXYZ, chemtools::CoordsXYZ> {lower_bnd, upper_bnd};
}


chemtools::UniformGrid chemtools::get_grid_from_coordinates_charges(
    const double* coordinates, const int* charges, int natoms, double add_amount
){

  py::array_t<double> py_coordinates(natoms * 3, coordinates);
  py::array_t<int> py_charges(natoms, charges);

  auto locals =
      py::dict("coordinates"_a = py_coordinates, "charges"_a = py_charges, "natoms"_a = natoms,
               "extension"_a = add_amount);
  py::exec(R"(
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    coordinates_arr = np.array(coordinates)
    coordinates_arr = np.reshape(coordinates, (natoms, 3), order="C")
    charges_arr = np.array(charges)

    # calculate center of mass of the nuclear charges:
    totz = np.sum(charges_arr)
    com = np.dot(charges_arr, coordinates_arr) / totz

    # calculate moment of inertia tensor:
    itensor = np.zeros([3, 3])
    for i in range(charges_arr.shape[0]):
        xyz = coordinates_arr[i] - com
        r = np.linalg.norm(xyz)**2.0
        tempitens = np.diag([r, r, r])
        tempitens -= np.outer(xyz.T, xyz)
        itensor += charges_arr[i] * tempitens

    _, v = np.linalg.eigh(itensor)  # v is a rotation matrix.
    new_coordinates = np.dot((coordinates_arr), v)  # Center the coordinate system then rotate it.

    all_l_bnds = []
    all_u_bnds = []
    for i in range(0, natoms):
      coords = new_coordinates[i].copy()
      charge_diff = 0
      if (charges_arr[i] == 1):
        charge_diff = 2.6
      elif (charges_arr[i] <= 6):
        charge_diff = 2.8
      elif (charges_arr[i] == 7):
        charge_diff = 3.0
      elif (charges_arr[i] == 8):
        charge_diff = 3.3
      elif (charges_arr[i] == 9):
        charge_diff = 2.9
      elif (charges_arr[i] <= 15):
        charge_diff = 3.3
      elif (charges_arr[i] == 16):
        charge_diff = 3.4
      elif (charges_arr[i] == 17):
        charge_diff = 3.5
      elif (charges_arr[i] <= 35):
        charge_diff = 3.5
      elif (charges_arr[i] <= 51):
        charge_diff = 3.7
      elif (charges_arr[i] == 52):
        charge_diff = 3.7
      elif (charges_arr[i] == 53):
        charge_diff = 3.8
      all_l_bnds.append(coords - charge_diff)
      all_u_bnds.append(coords + charge_diff)

    max_coordinate = np.amax(np.array(all_u_bnds), axis=0) + extension
    min_coordinate = np.amin(np.array(all_l_bnds), axis=0) - extension

    #spacing = 0.75
    # Compute the required number of points along x, y, and z axis with the regular coordinates
    #shape = np.abs(max_coordinate - min_coordinate) / spacing
    #shape = np.ceil(shape)
    #shape = np.array(shape, int)
    #print("shape", shape)

    min_coordinate = min_coordinate.dot(v.T)
    max_coordinate = max_coordinate.dot(v.T)
    
    #points = np.zeros((np.prod(shape), 3))
    #coords = np.array(
    #    np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    #)
    #coords = np.swapaxes(coords, 1, 2)
    #coords = coords.reshape(3, -1)
    #coords = coords.T
    #points = coords.dot(v.T) * spacing
    ## Compute coordinates of grid points relative to the origin
    #points += min_coordinate

    #axes = v.T * spacing
    #fig = plt.figure()
    #ax = plt.axes(projection="3d")
    #ax.scatter3D(coordinates_arr[:, 0], coordinates_arr[:, 1], coordinates_arr[:, 2], color="r")
    #ax.scatter3D(new_coordinates[:, 0], new_coordinates[:, 1], new_coordinates[:, 2], color="g")
    #ax.scatter3D(com[0], com[1], com[2])
    #ax.scatter3D(min_coordinate[0], min_coordinate[1], min_coordinate[2], color="k", s=50)
    #ax.scatter3D(max_coordinate[0], max_coordinate[1], max_coordinate[2],  color="k", s=50)
    #ax.scatter3D(min_coordinate[0] + axes[:, 0] / spacing,
    #             min_coordinate[1] + axes[:, 1]  / spacing,
    #             min_coordinate[2] + axes[:, 2] / spacing, cmap="Yellow")
    #ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], cmap="Yellow", alpha=0.25)
    #plt.show()

    # Test that all coordinates are within the grid.

    # Compute origin away from the grid, in the direction of axes
    origin = min_coordinate
    axes = v.T
    #print(axes)
  )", py::globals(), locals);

  // Grab the objects from Python
  py::array_t<double, py::array::f_style | py::array::forcecast>
      py_axes = locals["axes"].cast<py::array_t<double, py::array::f_style | py::array::forcecast>>();
  py::array_t<double, py::array::c_style | py::array::forcecast>
      py_origin = locals["min_coordinate"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  py::array_t<double, py::array::c_style | py::array::forcecast>
      py_max_coordinate = locals["max_coordinate"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

  // Convert them to std::array see utils.h
  chemtools::CoordsXYZ origin = {py_origin.at(0), py_origin.at(1), py_origin.at(2)};
  chemtools::CoordsXYZ max_coord = {py_max_coordinate.at(0), py_max_coordinate.at(1), py_max_coordinate.at(2)};
  chemtools::ThreeDMatrixColOrder axes = {
      py_axes.at(0, 0),
      py_axes.at(1, 0), py_axes.at(2, 0), py_axes.at(0, 1), py_axes.at(1, 1), py_axes.at(2, 1), py_axes.at(0, 2),
      py_axes.at(1, 2), py_axes.at(2, 2)
  };
  return chemtools::UniformGrid(origin, max_coord, axes);  //  std::make_tuple(origin, max_coord, axes);
}


