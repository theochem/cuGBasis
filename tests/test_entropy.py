import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
from test_utils import FCHK_FILES_FOR_TESTING
import pytest


@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_shannon_entropy_of_electron_density_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)
    spin = np.random.choice(["ab", "a", "b"], size=1)[0]

    grid_pts = np.random.uniform(-2, 2, size=(1000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_shannon = mol.compute_shannon_information_density(grid_pts, spin)

    mol2 = Molecule.from_file(fchk)
    density = mol2.compute_density(grid_pts, spin)
    err = np.abs(-np.log(density) * density - gpu_shannon)
    print(np.mean(err), np.max(err))
    assert np.all(err < 1e-8)
