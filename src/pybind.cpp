
#include <pybind11/pybind11.h>

#include "../include/pymolecule.cuh"
#include <cuda_runtime.h>

namespace py = pybind11;


PYBIND11_MODULE(gbasis_cuda, m) {
  m.doc() = "Test of documentation";

  py::class_<gbasis::Molecule>(m, "Molecule")
      .def(py::init<const std::string &>())
      //.def("compute_electron_density", &gbasis::Molecule::compute_electron_density)
      .def("basis_set_to_constant_memory",
        &gbasis::Molecule::basis_set_to_constant_memory,
        "Read a FCHK File and transfer it to constant memory. Required for all calculations"
        "If do_decontracted_basis=True, then basis-set is stored as segmented contraction shell",
        py::arg("do_segmented_basis") = false
      )
      .def("compute_electron_density",
           &gbasis::Molecule::compute_electron_density,
           "Compute electron density."
           )
      .def("compute_electron_density_on_cubic_grid",
           &gbasis::Molecule::compute_electron_density_cubic,
           "Compute electron density on a cubic grid. Memory-efficient."
      )
      .def("compute_electron_density_gradient",
           &gbasis::Molecule::compute_electron_density_gradient,
           "Compute gradient of electron density."
      )
      .def("compute_laplacian_electron_density",
           &gbasis::Molecule::compute_laplacian,
           "Compute the Laplacian of the electron density."
      )
      .def("compute_positive_definite_kinetic_energy_density",
           &gbasis::Molecule::compute_positive_definite_kinetic_energy,
           "Compute the positive definite kinetic energy density."
      )
      .def("compute_general_kinetic_energy_density",
           &gbasis::Molecule::compute_positive_definite_kinetic_energy,
           "Compute the general kinetic energy density."
      )
      .def("compute_norm_of_vector",
           &gbasis::Molecule::compute_norm_of_vector,
           "Compute the norm of a vector."
      )
      .def("compute_reduced_density_gradient",
           &gbasis::Molecule::compute_reduced_density_gradient,
           "Compute the reduced density gradient."
      )
      .def("compute_weizsacker_ked",
           &gbasis::Molecule::compute_weizsacker_ked,
           "Compute the Weizsacker Kinetic energy density."
      )
      .def("compute_thomas_fermi_ked",
           &gbasis::Molecule::compute_thomas_fermi_ked,
           "Compute the Thomas-Fermi kinetic energy density."
      )
      .def("compute_general_gradient_expansion_ked",
           &gbasis::Molecule::compute_general_gradient_expansion_ked,
           "Compute the general gradient expansion approximation of kinetic energy density."
      )
      .def("compute_gradient_expansion_ked",
           &gbasis::Molecule::compute_gradient_expansion_ked,
           "Compute the gradient expansion approximation of kinetic energy density."
      )
      .def("compute_empirical_gradient_expansion_ked",
           &gbasis::Molecule::compute_empirical_gradient_expansion_ked,
           "Compute the empirical gradient expansion approximation of kinetic energy density."
      )
      .def("compute_general_ked",
           &gbasis::Molecule::compute_general_ked,
           "Compute the general(ish) kinetic energy density."
      )
      .def("compute_hamiltonian_ked",
           &gbasis::Molecule::compute_hamiltonian_ked,
           "Compute the Hamiltonian kinetic energy density."
      )
      .def("compute_electrostatic_potential",
           &gbasis::Molecule::compute_electrostatic_potential,
           "Compute electrostatic potential. "
           "Basis-set needs to be segmented, i.e. do_segmented_Basis=True must be set to True in "
           "`basis_set_to_constant_memory`."
      )
      .def("get_file_path", &gbasis::Molecule::getFilePath);

}