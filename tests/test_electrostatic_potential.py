import numpy as np
import gbasis_cuda
from chemtools.wrappers import Molecule
import pytest


@pytest.mark.parametrize("fchk",
                         [
                             "atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
                             "atom_he.fchk",
                             "atom_be.fchk",
                             "h2o.fchk",
                             "ch4.fchk",
                             "qm9_000092_HF_cc-pVDZ.fchk"
                         ]
                         )
def test_electrostatic_potential_against_horton(fchk):
    fchk = "./tests/data/" + fchk
    mol = gbasis_cuda.Molecule(fchk)
    mol.basis_set_to_constant_memory(do_segmented_basis=True)
    grid_pts = np.random.uniform(-2, 2, size=(2000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_esp = mol.compute_electrostatic_potential(grid_pts)
    mol2 = Molecule.from_file(fchk)
    cpu_esp = mol2.compute_esp(grid_pts)
    assert np.all(np.abs(cpu_esp - gpu_esp) < 1e-8)
