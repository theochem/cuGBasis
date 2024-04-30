import numpy as np
import cugbasis
from grid.cubic import UniformGrid
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
def test_electron_density_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-2, 2, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_gradient = mol.compute_density(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_gradient = mol2.compute_density(grid_pts)

    assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)


# @pytest.mark.parametrize("fchk",
#                          [
#                              "./tests/data/E948_rwB97XD_def2SVP.fchk",
#                              "./tests/data/h2o.fchk",
#                              "./tests/data/ch4.fchk",
#                              "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
#                          ]
#                          )
# def test_electron_density_on_cubic_grid_against_horton(fchk):
#     mol = cugbasis.Molecule(fchk)
#     mol2 = Molecule.from_file(fchk)
#
#     grid = UniformGrid.from_molecule(mol2.numbers, mol2.coordinates, spacing=0.5)
#     gpu_density = mol.compute_density_on_cubic_grid(grid.origin, grid.axes, grid.shape, False)
#     cpu_density = mol2.compute_density(grid.points)
#     assert np.all(np.abs(cpu_density - gpu_density) < 1e-8)
