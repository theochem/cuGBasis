import numpy as np
import cugbasis
import pytest
from iodata import load_one
from scipy.special import erf
from grid.molgrid import MolGrid
from grid.hirshfeld import HirshfeldWeights
from chemtools.wrappers import Molecule

class Promolecular:
    def __init__(self, atom_nums, mol_coords):
        self.atom_nums = atom_nums
        self.mol_coords = mol_coords
        # These have shape (M, K_i) where M is the number of atoms.
        self.coeffs_s, self.coeffs_p, self.exps_s, self.exps_p = self.load_coefficients_exponents()

    @property
    def numb_atoms(self):
        return len(self.mol_coords)

    def load_coefficients_exponents(self):
        all_results = np.load("./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")
        coeffs_s, coeffs_p = [], []
        exps_s, exps_p = [], []

        atoms = np.array(["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
                          "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca",
                          "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu",
                          "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr",
                          "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag",
                          "cd", "in", "sn", "sb", "te", "i", "xe"])
        atomic_numbs = np.array([1 + i for i in range(0, len(atoms))])
        atoms_id = [atoms[np.where(atomic_numbs == int(x))[0]][0] for x in self.atom_nums]
        for atom in atoms_id:
            coeffs = all_results[atom.capitalize() + "_coeffs_s"]
            exps = all_results[atom.capitalize()  + "_exps_s"]
            coeffs_s.append(coeffs)
            exps_s.append(exps)

            coeffs = all_results[atom.capitalize()  + "_coeffs_p"]
            exps = all_results[atom.capitalize()  + "_exps_p"]
            coeffs_p.append(coeffs)
            exps_p.append(exps)
        return coeffs_s, coeffs_p, exps_s, exps_p

    def compute_density(self, pts):
        r"""


        Notes
        -----
        Approach One - For-loop over atom


        """
        density = np.zeros(len(pts), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # Center the points to the atom
            centered_pts = np.sum((pts - center)**2.0, axis=1)
            which_to_do = centered_pts < 10.0**2.0  # a.u.
            centered_pts = centered_pts[which_to_do]

            # Has shape (K, N), where K number of exponentials and N number of points
            exponential = np.exp(-exps[:, None] * centered_pts)
            normalization = (exps / np.pi)**1.5
            # Calculate S-type Gaussians
            density[which_to_do] += np.einsum(
                "i,i,ij->j", coeffs, normalization, exponential, optimize=True
            )

            # Calculate P-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            # print(coeffs)
            # print(exps)
            if len(coeffs) != 0:
                exponential = np.exp(-exps[:, None] * centered_pts)
                normalization = (2.0 * exps**2.5) / (3.0 * np.pi**1.5)
                # normalization = np.ones(normalization.shape)
                density[which_to_do] += np.einsum(
                    "i,i,j,ij->j",
                    coeffs, normalization, centered_pts, exponential, optimize=True
                )
        return density

    def compute_esp(self, pts):
        esp = np.zeros(len(pts), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]

            # Calculate ESP of promolecular of s-type Gaussians
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # let p = (a/pi)^1.5 e^(-a r_C^2)
            #  then the esp integral: \int p(r) / |r - P|^2 = erf(sqrt(a) R_{PC}) / R_{PC}
            r_pc = np.linalg.norm(pts - center, axis=1)
            erf_func = erf(np.sqrt(exps[:, None]) * r_pc)
            esp += np.einsum("i,ij,j->j", coeffs, erf_func, 1.0 / r_pc, optimize=True)

            # Calculate ESP of promolecular of p-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            if len(coeffs) != 0:
                etf_func = erf(np.sqrt(exps[:, None]) * r_pc) / r_pc
                exponential = np.exp(-exps[:, None] * r_pc**2.0)
                normalization = (2.0 * exps ** 2.5) / (3.0 * np.pi ** 1.5)
                # \int r^2 e^{-a r^2} / |r - P| = N^{-1} etf(sqrt(a) R_{PC}) / R_{PC} - pi e^{-a R_{PC}^2} / a^2
                #   where N is the normalization constant above
                integral = etf_func - normalization[:, None] * np.pi * exponential / exps[:, None]**2.0
                esp += np.einsum("i,ij->j", coeffs, integral, optimize=True)
        centered = np.linalg.norm(self.mol_coords[:, None, :] - pts[None, :, :], axis=2)
        # i is number of atoms,    j is the points,
        return np.einsum("i,ij->j", self.atom_nums, 1.0 / centered) - esp


@pytest.mark.parametrize("molecule", [
    "./tests/data/atom_o.fchk",
    "./tests/data/atom_c_g_pure_orbital.fchk",
    "./tests/data/h2o.fchk",
    "./tests/data/ch4.fchk",
    "./tests/data/test.fchk",
    "./tests/data/test2.fchk",
    "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
])
def test_promolecular_density(molecule):
    mol_iodata = load_one(molecule)
    atcoords = mol_iodata.atcoords
    atnums = mol_iodata.atnums

    promol_gpu = cugbasis.Promolecule(atcoords, atnums, len(atnums),
                                      "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")
    promol_cpu = Promolecular(atnums, atcoords)

    random_pts = np.random.uniform(low=-1, high=1, size=(10000, 3)) + np.mean(atcoords, axis=0)
    density_gpu = promol_gpu.compute_density(random_pts)
    density_cpu = promol_cpu.compute_density(random_pts)

    assert np.all(np.abs(density_gpu - density_cpu) < 1e-8)


@pytest.mark.parametrize("molecule", [
    "./tests/data/atom_o.fchk",
    "./tests/data/atom_c_g_pure_orbital.fchk",
    "./tests/data/h2o.fchk",
    "./tests/data/ch4.fchk",
    "./tests/data/test.fchk",
    "./tests/data/test2.fchk",
    "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
])
def test_atomic_promolecular_densities_on_variety_of_systems(molecule):
    mol_iodata = load_one(molecule)
    atcoords = mol_iodata.atcoords
    atnums = mol_iodata.atnums

    promol_gpu = cugbasis.Promolecule(atcoords, atnums, len(atnums),
                                      "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")

    random_pts = np.random.uniform(low=-1, high=1, size=(100000, 3)) + np.mean(atcoords, axis=0)
    atomic_density = promol_gpu.compute_atomic_densities(random_pts)
    promol_density = promol_gpu.compute_density(random_pts)
    assert np.all(np.abs(np.sum(atomic_density, axis=1) - promol_density) < 1e-8)


def test_atomic_promolecular_with_many_atoms():
    SIZE = 10000
    atcoords = np.zeros((SIZE, 3))
    atcoords[:, 0] = np.arange(1, SIZE + 1)
    atnums = np.ones(atcoords.shape[0], dtype=int)
    promol_gpu =  cugbasis.Promolecule(atcoords, atnums, len(atnums),
                                       "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")

    random_pts = np.random.uniform(0.0, SIZE, size=(SIZE, 3))
    random_pts[:, 1] = 0.0
    random_pts[:, 2] = 0.0
    random_pts[:, 0] = np.linspace(1, SIZE + 1, num=SIZE)
    atomic_density = promol_gpu.compute_atomic_densities(random_pts)
    promol_density = promol_gpu.compute_density(random_pts)
    sum_promol = np.sum(atomic_density, axis=1)
    assert np.all(np.abs(sum_promol - promol_density) < 1e-6)


def test_promolecular_integral_with_interpolation_params():
    SIZE = 10
    atcoords = np.zeros((SIZE, 3))
    atcoords[:, 0] = np.arange(1, SIZE + 1)
    atnums = np.ones(atcoords.shape[0], dtype=int)
    promol =  cugbasis.Promolecule(atcoords, atnums, len(atnums),
                                   "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")
    mol_grid = MolGrid.from_preset(
        atnums=atnums,
        atcoords=atcoords,
        preset="coarse",
        aim_weights=HirshfeldWeights(),
        store=True,
    )

    interpolation_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    density = promol.compute_density(mol_grid.points, interpolation_params)
    assert np.abs(mol_grid.integrate(density) - np.sum(interpolation_params)) < 1e-3


def test_promolecular_with_many_atoms():
    SIZE = 10000
    atcoords = np.zeros((SIZE, 3))
    atcoords[:, 0] = np.arange(1, SIZE + 1)
    atnums = np.ones(atcoords.shape[0], dtype=int)
    promol_gpu =  cugbasis.Promolecule(atcoords, atnums, len(atnums),
                                   "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")
    promol_cpu =  Promolecular(atnums, atcoords)

    random_pts = np.random.uniform(0.0, SIZE, size=(10000, 3))
    random_pts[:, 1] = 0.0
    random_pts[:, 2] = 0.0
    dens_gpu = promol_gpu.compute_density(random_pts)
    dens_cpu = promol_cpu.compute_density(random_pts)
    assert np.all(np.abs(dens_gpu - dens_cpu) < 1e-7)


@pytest.mark.parametrize("molecule", [
    "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/atom_o.fchk",
    "./tests/data/atom_c_g_pure_orbital.fchk",
    "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/h2o.fchk",
    "./tests/data/ch4.fchk",
])
def test_promolecular_esp(molecule):
    mol_iodata = load_one(molecule)
    atcoords = mol_iodata.atcoords
    atnums = mol_iodata.atnums

    promol_gpu = cugbasis.Promolecule(atcoords, atnums, len(atnums), "./data/result_kl_fpi_method_cugbasis_atomdb_hci_slater.npz")
    promol_cpu = Promolecular(atnums, atcoords)

    random_pts = np.random.uniform(low=-1, high=1, size=(10000, 3)) + np.mean(atcoords, axis=0)
    esp_gpu = promol_gpu.compute_esp(random_pts)
    esp_cpu = promol_cpu.compute_esp(random_pts)

    assert np.all(np.abs(esp_gpu - esp_cpu) < 1e-8)
