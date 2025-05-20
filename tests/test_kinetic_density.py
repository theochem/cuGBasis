import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest
from test_utils import FCHK_FILES_FOR_TESTING

@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_positive_definite_of_electron_density_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)
    spin = np.random.choice(["ab", "a", "b"], size=1)[0]

    grid_pts = np.random.uniform(-2, 2, size=(1000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_kinetic = mol.compute_positive_definite_kinetic_energy_density(grid_pts, spin)

    mol2 = Molecule.from_file(fchk)
    cpu_kinetic = mol2.compute_ked(grid_pts, spin)

    assert np.all(np.abs(cpu_kinetic - gpu_kinetic) < 1e-8)
