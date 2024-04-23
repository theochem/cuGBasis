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
def test_hessian_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(1000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")

    mol2 = Molecule.from_file(fchk)
    cpu_hessian = mol2.compute_hessian(grid_pts)
    gpu_hessian = mol.compute_hessian(grid_pts)

    assert np.all(np.abs(gpu_hessian - cpu_hessian) < 1e-8)
