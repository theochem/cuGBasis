import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest
from test_utils import FCHK_FILES_FOR_TESTING

@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_hessian_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(20000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")

    mol2 = Molecule.from_file(fchk)
    cpu_hessian = mol2.compute_hessian(grid_pts)
    gpu_hessian = mol.compute_hessian(grid_pts)

    assert np.all(np.abs(gpu_hessian - cpu_hessian) < 1e-8)
