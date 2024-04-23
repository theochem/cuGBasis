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
def test_gradient_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    # gpu_gradient = mol.compute_electron_density_gradient(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_gradient = mol2.compute_gradient(grid_pts)

    # assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)

    for i, pt in enumerate(grid_pts):
        print("PT ", pt)
        print("CPU ", cpu_gradient[i])
        gpu = mol.compute_gradient(np.array([pt]))[0]
        print("GPU ", gpu)
        print("Error ", np.abs(cpu_gradient[i] - gpu))
        print("")
        assert np.all(np.abs(cpu_gradient[i] - gpu) < 1e-8)



