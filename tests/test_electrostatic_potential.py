import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest


@pytest.mark.parametrize("fchk",
                         [
                             "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
                             "./tests/data/atom_he.fchk",
                             "./tests/data/atom_be.fchk",
                             "./tests/data/h2o.fchk",
                             "./tests/data/ch4.fchk",
                             "./tests/data/qm9_000092_HF_cc-pVDZ.fchk"
                         ]
                         )
def test_electrostatic_potential_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)
    grid_pts = np.random.uniform(-2, 2, size=(2000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_esp = mol.compute_electrostatic_potential(grid_pts)
    mol2 = Molecule.from_file(fchk)
    cpu_esp = mol2.compute_esp(grid_pts)
    assert np.all(np.abs(cpu_esp - gpu_esp) < 1e-8)
