import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest


@pytest.mark.parametrize("fchk",
                         [
                             "./tests/data/E948_rwB97XD_def2SVP.fchk",
                             "./tests/data/h2o.fchk",
                             "./tests/data/ch4.fchk",
                             "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
                         ]
                         )
def test_positive_definite_of_electron_density_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-2, 2, size=(1000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_kinetic = mol.compute_positive_definite_kinetic_energy_density(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_kinetic = mol2.compute_ked(grid_pts)

    assert np.all(np.abs(cpu_kinetic - gpu_kinetic) < 1e-8)
