import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest
from test_utils import FCHK_FILES_FOR_TESTING

@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_laplacian_of_electron_density_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-2, 2, size=(1000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_laplacian = mol.compute_laplacian(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_laplacian = mol2.compute_laplacian(grid_pts)

    assert np.all(np.abs(cpu_laplacian - gpu_laplacian) < 1e-8)
