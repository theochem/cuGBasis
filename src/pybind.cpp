
#include <pybind11/pybind11.h>

#include "../include/pymolecule.cuh"
#include <cuda_runtime.h>

namespace py = pybind11;


PYBIND11_MODULE(chemtools_cuda, m) {
  m.doc() = "Test of documentation";

  py::class_<chemtools::Molecule>(m, "Molecule")
      .def(py::init<const std::string &>())
      // DEPRECIATED because now the molecule itself reads off the basis-set.
//      .def("basis_set_to_constant_memory",
//        &chemtools::Molecule::basis_set_to_constant_memory,
//        "Read a FCHK File and transfer it to constant memory. Required for all calculations"
//        "If do_decontracted_basis=True, then basis-set is stored as segmented contraction shell",
//        py::arg("do_segmented_basis") = false
//      )
      .def("compute_electron_density",
           &chemtools::Molecule::compute_electron_density,
           py::return_value_policy::reference_internal,
           "Compute electron density."
           )
      .def("compute_electron_density_on_cubic_grid",
           &chemtools::Molecule::compute_electron_density_cubic,
           // TensorMap<> policies for return_value_policy returns an error is an rvalue
           py::return_value_policy::move,
           "Compute electron density on a cubic grid. Memory-efficient."
      )
      .def("compute_electron_density_gradient",
           &chemtools::Molecule::compute_electron_density_gradient,
           py::return_value_policy::reference_internal,
           "Compute gradient of electron density."
      )
      .def("compute_electron_density_hessian",
           &chemtools::Molecule::compute_electron_density_hessian,
           py::return_value_policy::move,
           "Compute Hessian of electron density."
      )
      .def("compute_laplacian_electron_density",
           &chemtools::Molecule::compute_laplacian,
           py::return_value_policy::reference_internal,
           "Compute the Laplacian of the electron density."
      )
      .def("compute_positive_definite_kinetic_energy_density",
           &chemtools::Molecule::compute_positive_definite_kinetic_energy,
           py::return_value_policy::reference_internal,
           "Compute the positive definite kinetic energy density."
      )
      .def("compute_general_kinetic_energy_density",
           &chemtools::Molecule::compute_general_ked,
           py::return_value_policy::reference_internal,
           "Compute the general kinetic energy density."
      )
      .def("compute_norm_of_vector",
           &chemtools::Molecule::compute_norm_of_vector,
           py::return_value_policy::reference_internal,
           "Compute the norm of a vector."
      )
      .def("compute_reduced_density_gradient",
           &chemtools::Molecule::compute_reduced_density_gradient,
           py::return_value_policy::reference_internal,
           "Compute the reduced density gradient."
      )
      .def("compute_weizsacker_ked",
           &chemtools::Molecule::compute_weizsacker_ked,
           py::return_value_policy::reference_internal,
           "Compute the Weizsacker Kinetic energy density."
      )
      .def("compute_thomas_fermi_ked",
           &chemtools::Molecule::compute_thomas_fermi_ked,
           py::return_value_policy::reference_internal,
           "Compute the Thomas-Fermi kinetic energy density."
      )
      .def("compute_general_gradient_expansion_ked",
           &chemtools::Molecule::compute_general_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           "Compute the general gradient expansion approximation of kinetic energy density."
      )
      .def("compute_gradient_expansion_ked",
           &chemtools::Molecule::compute_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           "Compute the gradient expansion approximation of kinetic energy density."
      )
      .def("compute_empirical_gradient_expansion_ked",
           &chemtools::Molecule::compute_empirical_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           "Compute the empirical gradient expansion approximation of kinetic energy density."
      )
      .def("compute_general_ked",
           &chemtools::Molecule::compute_general_ked,
           py::return_value_policy::reference_internal,
           "Compute the general(ish) kinetic energy density."
      )
      .def("compute_hamiltonian_ked",
           &chemtools::Molecule::compute_hamiltonian_ked,
           py::return_value_policy::reference_internal,
           "Compute the Hamiltonian kinetic energy density."
      )
      .def("compute_shannon_information_density",
           &chemtools::Molecule::compute_shannon_information_density,
           py::return_value_policy::reference_internal,
           "Compute the Shannon information density."
      )
      .def("compute_electrostatic_potential",
           &chemtools::Molecule::compute_electrostatic_potential,
           py::return_value_policy::reference_internal,
           "Compute electrostatic potential. "
           "Basis-set needs to be segmented, i.e. do_segmented_Basis=True must be set to True in "
           "`basis_set_to_constant_memory`."
      )
      // Coordinates
      .def_property_readonly(
          "coordinates", &chemtools::Molecule::getCoordinates, py::return_value_policy::reference_internal
          )
      .def_property_readonly(
          "numbers", &chemtools::Molecule::getNumbers, py::return_value_policy::reference_internal
      )
      .def("get_file_path", &chemtools::Molecule::getFilePath);
}