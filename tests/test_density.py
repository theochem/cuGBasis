import numpy as np
import cugbasis
from grid.cubic import UniformGrid
from chemtools.wrappers import Molecule
import pytest

mol = cugbasis.Molecule("./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk")

grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
gpu_gradient = mol.compute_density(grid_pts)

mol2 = Molecule.from_file("./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk")
cpu_gradient = mol2.compute_density(grid_pts)

assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)

@pytest.mark.parametrize("fchk",
                         [
                             "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
                             "./tests/data/atom_he.fchk",
                             "./tests/data/atom_be.fchk",
                             "./tests/data/atom_be_f_pure_orbital.fchk",
                             "./tests/data/atom_be_f_cartesian_orbital.fchk",
                             "./tests/data/atom_kr.fchk",
                             "./tests/data/atom_o.fchk",
                             "./tests/data/atom_c_g_pure_orbital.fchk",
                             "./tests/data/atom_mg.fchk",
                             "./tests/data/E948_rwB97XD_def2SVP.fchk",
                             "./tests/data/test.fchk",
                             "./tests/data/test2.fchk",
                             "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
                             "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
                             "./tests/data/h2o.fchk",
                             "./tests/data/ch4.fchk",
                             "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
                             "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
                         ]
                         )
def test_electron_density_against_horton_small_molecules(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_gradient = mol.compute_density(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_gradient = mol2.compute_density(grid_pts)

    assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)
