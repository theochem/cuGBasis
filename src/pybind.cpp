
#include <pybind11/pybind11.h>

#include "../include/pymolecule.cuh"

namespace py = pybind11;


PYBIND11_MODULE(cugbasis, m) {
  m.doc() = "Molecule class tht reads wavefunction information and computes various quantities.";
  py::options options;
  options.disable_function_signatures();

  py::class_<chemtools::ProMolecule>(m, "Promolecule")
    .def(py::init(&chemtools::ProMolecule::create), R"pbdoc(Initialize the Promolecule class.

Parameters
----------
atom_coords: numpy(M,3)
    Atomic Coordinates (a.u.)
atom_nums: numpy(M,)
    Atomic Numbers
natoms: int
    Number of atoms M
path: str
    Path to the ".npz" containing promolecular coefficients and exponents obtained from BFit.
    This ".npz" is within the folder "./data/".
)pbdoc")
    .def("compute_density",
      &chemtools::ProMolecule::compute_electron_density,
      py::return_value_policy::reference_internal,
      R"pbdoc(Compute promolecular density :math:`\rho^\circ(\mathbf{r})`.

          .. math::
            \rho^\circ(\mathbf{r}) = \sum_A \bigg[ \sum_i c_i \frac{\alpha}{\pi}^{3/2} e^{-\alpha r_{A} ^2} +
            \sum_j d_j \frac{2 \alpha^{5/2}}{3 \pi^{3/2}} r_A^2 e^{-\alpha r_A^2} \bigg],

          where :math:`A` is an atom, :math:`c_i, d_j` are the s-type, p-type coefficients respectively.
          Note this is normalized. Please see the paper "An information-theoretic approach to basis-set fitting of
          electron densities and other non-negative functions".

          Parameters
          ----------
          points: ndarray(N, 3)
              Cartesian coordinates of :math:`N` points in three-dimensions.

          Returns
          -------
          ndarray(N,)
              The promolecular density evaluated on each point.
      )pbdoc"
    )
    .def("compute_esp",
    &chemtools::ProMolecule::compute_electrostatic_potential,
    py::return_value_policy::reference_internal,
    R"pbdoc(Compute the electrostatic potential on :math:`\rho^\circ(\mathbf{r})`.

              .. math::
                \Phi(\mathbf{r}) = \sum_{i=1}^{N_{atom}} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}  -
                 \int \frac{\rho^\circ(\mathbf{r}^\prime)}{|\mathbf{r} - \mathbf{r}^\prime| }d\mathbf{r}^\prime,

              where :math:`Z_A` is the atomic number of atom A.

              Parameters
              ----------
              points: ndarray(N, 3)
                  Cartesian coordinates of :math:`N` points in three-dimensions.

              Returns
              -------
              ndarray(N,)
                  The promolecular electrostatic potential evaluated on each point.
          )pbdoc"
    )
    .def_property_readonly(
      "atcoords", &chemtools::ProMolecule::GetCoordAtoms, py::return_value_policy::reference_internal,
      "Cartesian coordinates of the atomic centers."
    )
    .def_property_readonly(
      "atnums", &chemtools::ProMolecule::GetAtomicNumbers, py::return_value_policy::reference_internal,
      "Atomic number of atomic centers."
    )
    .def_property_readonly(
    "promol_coeffs", &chemtools::ProMolecule::GetPromolCoefficients, py::return_value_policy::reference_internal,
    "Get the coefficients as a dictionary with keys element_parameter_type (e.g. `f_coeffs_p`)."
    )
    .def_property_readonly(
    "promol_exps", &chemtools::ProMolecule::GetPromolExponents, py::return_value_policy::reference_internal,
    "Get the exponents as a dictionary with keys element_parameter_type (e.g. `f_coeffs_p`)."
    );

  py::class_<chemtools::Molecule>(m, "Molecule")
      .def(py::init<const std::string &>(), R"pbdoc(Initialize the molecule class.

    Parameters
    ----------
    file_path: str
        Wavefunction file path.
)pbdoc")
      .def("compute_density",
           &chemtools::Molecule::compute_electron_density,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute electron density :math:`\rho(\mathbf{r})`.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

    Returns
    -------
    ndarray(N,)
        The electron density evaluated on each point.
)pbdoc"
           )
      .def("compute_molecular_orbitals",
           &chemtools::Molecule::compute_molecular_orbitals,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute Molecular Orbitals

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

    Returns
    -------
    ndarray(M, N)
        The M molecular orbitals evaluated on each point.
)pbdoc"
      )
      .def("compute_gradient",
           &chemtools::Molecule::compute_electron_density_gradient,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the gradient of the electron density.

    .. math::
         \nabla \rho(\mathbf{r}) = \bigg(\frac{\partial \rho}{\partial x}, \frac{\partial \rho}{\partial y},
         \frac{\partial \rho}{\partial z} \bigg),

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

    Returns
    -------
    ndarray(N, 3)
        The gradient of the electron density evaluated on each point.
)pbdoc"
      )
      .def("compute_hessian",
           &chemtools::Molecule::compute_electron_density_hessian,
           py::return_value_policy::move,
           R"pbdoc(Compute the Hessian of the electron density.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

    Returns
    -------
    ndarray(N, 3, 3)
        The Hessian of the electron density evaluated on each point.
)pbdoc"
      )
      .def("compute_laplacian",
           &chemtools::Molecule::compute_laplacian,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the Laplacian of the electron density.

    .. math::
         \nabla^2 \rho(\mathbf{r}) = \sqrt{\bigg(\frac{\partial \rho}{\partial x} \bigg)^2 +
          \bigg(\frac{\partial \rho}{\partial y} \bigg)^2 + \bigg(\frac{\partial \rho}{\partial z} \bigg)^2}

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

)pbdoc"
      )
      .def("compute_positive_definite_kinetic_energy_density",
           &chemtools::Molecule::compute_positive_definite_kinetic_energy,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the positive definite kinetic energy density on a grid of points.

    .. math::
        \tau_{PD}(\mathbf{r}) = \frac{1}{2} \sum_{i=1}^{N} \big|\nabla \phi_i (\mathbf{r})\big|^2,

    where :math:`\phi_i` is the molecular orbitals. In the QTAIM community, this is denoted as
    :math:`G(\mathbf{r})`.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

)pbdoc"
      )
      .def("compute_general_kinetic_energy_density",
           &chemtools::Molecule::compute_general_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the general kinetic energy density on a grid of points.

    .. math::
        t_\alpha(r) = t_+(r) + \alpha \nabla^2 \rho(r)

    where :math:`t_+(r)` is the positive definite kinetic energy density and
    :math:`\nabla^2` is the Laplacian.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
    alpha: float
        Constant parameter.
)pbdoc"
      )
      .def("compute_norm_of_vector",
           &chemtools::Molecule::compute_norm_of_vector,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the norm of a three-dimensional vector using GPU.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

)pbdoc"
      )
      .def("compute_reduced_density_gradient",
           &chemtools::Molecule::compute_reduced_density_gradient,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the reduced density gradient.

    .. math::
         s(\mathbf{r}) = \frac{1}{2(3 \pi^2)^{1/3}}\frac{|\nabla \rho(\mathbf{r})|}{\rho(\mathbf{r})^{4/3}}.

    Note it may be more efficient to compute each component individually, then using
    Numpy to compute this quanity.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.

)pbdoc"
      )
      .def("compute_weizsacker_ked",
           &chemtools::Molecule::compute_weizsacker_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the Weizsacker Kinetic energy density.

    .. math::
         \tau_W(\mathbf{r}) = \frac{\big|\nabla \rho(\mathbf{r}) \big|^2}{8\rho(\mathbf{r})}

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_thomas_fermi_ked",
           &chemtools::Molecule::compute_thomas_fermi_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the Thomas-Fermi kinetic energy density.

    .. math::
        \tau_{TF}(\mathbf{r}) = \frac{3}{10} (3\pi)^{2/3} \rho(\mathbf{r})^{5/3}

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_general_gradient_expansion_ked",
           &chemtools::Molecule::compute_general_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the general gradient expansion approximation of kinetic energy density.

    .. math::
        \tau_{GGE}(\mathbf{r}) = \tau_{TF}(\mathbf{r}) + \alpha \tau_{W}(\mathbf{r}) + \beta \nabla^2 \rho(\mathbf{r})

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
    alpha: float
        Constant parameter.
    beta: float
        Constant parameter.
)pbdoc"
      )
      .def("compute_gradient_expansion_ked",
           &chemtools::Molecule::compute_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the gradient expansion approximation of kinetic energy density

    .. math::
        \tau_{GEA}(\mathbf{r}) = \tau_{TF}(\mathbf{r}) + \frac{1}{9} \tau_W(\mathbf{r}) +
        \frac{1}{6} \nabla^2 \rho(\mathbf{r})

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_empirical_gradient_expansion_ked",
           &chemtools::Molecule::compute_empirical_gradient_expansion_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the empirical gradient expansion approximation of kinetic energy density.

    .. math::
        \tau_{EGEA}(\mathbf{r}) = \tau_{TF}(\mathbf{r}) + \frac{1}{5} \tau_W(\mathbf{r}) +
        \frac{1}{6} \nabla^2 \rho(\mathbf{r})

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_general_ked",
           &chemtools::Molecule::compute_general_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the general(ish) kinetic energy density.

    .. math::
        \tau_G(\mathbf{r}, \alpha)  = \tau_{PD}(\mathbf{r}) + \frac{1}{4} (\alpha - 1) \nabla^2 \rho(\mathbf{r}),

    where :math:`a` is a constant parameter.  When :math:`a=0`, then it is the 'Schrödinger' kinetic energy, when
    :math:`a=1`, then it is the positive-definite kinetic energy.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
    alpha: float
        The constant parameter.
)pbdoc"
      )
      .def("compute_schrodinger_ked",
           &chemtools::Molecule::compute_hamiltonian_ked,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the Schrödinger kinetic energy density.

    This is the general(sh) kinetic energy density when :math:`a=0`:

    .. math::
        K(\mathbf{r})  = \tau_{PD}(\mathbf{r}) - \frac{1}{4} \nabla^2 \rho(\mathbf{r}),

    In the QTAIM community, this is denoted as :math:`K(\mathbf{r})`.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_shannon_information_density",
           &chemtools::Molecule::compute_shannon_information_density,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the Shannon information density."

    .. math::
         s(\mathbf{r}) = -\rho(\mathbf{r}) \log \rho(\mathbf{r}),

    where the integral gives the Shannon entropy of the electron density.

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      .def("compute_electrostatic_potential",
           &chemtools::Molecule::compute_electrostatic_potential,
           py::return_value_policy::reference_internal,
           R"pbdoc(Compute the molecular electrostatic potential.

    .. math::
        \Phi(\mathbf{r}) = \sum_{i=1}^{N_{atom}} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}  -
         \int \frac{\rho(\mathbf{r}^\prime)}{|\mathbf{r} - \mathbf{r}^\prime| }d\mathbf{r}^\prime,

    Parameters
    ----------
    points: ndarray(N, 3)
        Cartesian coordinates of :math:`N` points in three-dimensions.
)pbdoc"
      )
      // Properties
      .def_property_readonly(
          "atcoords", &chemtools::Molecule::getCoordinates, py::return_value_policy::reference_internal,
          "Cartesian coordinates of the atomic centers."
          )
      .def_property_readonly(
          "atnums", &chemtools::Molecule::getNumbers, py::return_value_policy::reference_internal,
          "Atomic number of atomic centers."
      )
      .def("get_file_path", &chemtools::Molecule::getFilePath, "Get the wavefunction file path.");
}