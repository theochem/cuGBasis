import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest

@pytest.mark.parametrize("fchk", [
                                      "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
                                      "./tests/data/atom_be.fchk",
                                      "./tests/data/atom_be_f_pure_orbital.fchk",
                                      "./tests/data/atom_be_f_cartesian_orbital.fchk",
                                      "./tests/data/atom_kr.fchk",
                                      "./tests/data/atom_o.fchk",
                                      "./tests/data/atom_c_g_pure_orbital.fchk",
                                      "./tests/data/atom_mg.fchk",
                                      "./tests/data/test.fchk",
                                      "./tests/data/test2.fchk",
                                      "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
                                      "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
                                      "./tests/data/h2o.fchk",
                                      "./tests/data/ch4.fchk",
                                      "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
                                      "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
                                      ])
def test_molecular_orbitals_against_horton(fchk):
    spin = np.random.choice(["a", "b"], size=1)[0]
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_mo = mol.compute_molecular_orbitals(grid_pts, spin)

    mol2 = Molecule.from_file(fchk)
    cpu_mo = mol2.compute_molecular_orbital(grid_pts, spin, index=list(range(1, mol2.mo.occupation[0].shape[0] + 1)))
    assert np.all(np.abs(cpu_mo - gpu_mo.T) < 1e-8)


@pytest.mark.parametrize("fchk", [
    "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/atom_be.fchk",
    "./tests/data/atom_be_f_pure_orbital.fchk",
    "./tests/data/atom_be_f_cartesian_orbital.fchk",
    "./tests/data/atom_kr.fchk",
    "./tests/data/atom_o.fchk",
    "./tests/data/atom_c_g_pure_orbital.fchk",
    "./tests/data/atom_mg.fchk",
    "./tests/data/test.fchk",
    "./tests/data/test2.fchk",
    "./tests/data/atom_08_O_N08_M3_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/h2o.fchk",
    "./tests/data/ch4.fchk",
    "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
    "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
])
def test_occ_attribute_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)
    mol2 = Molecule.from_file(fchk)

    assert np.all(np.abs(mol.occs_a - mol2.mo.occupation[0]) < 1e-8)
    assert np.all(np.abs(mol.occs_b - mol2.mo.occupation[1]) < 1e-8)

    homo_a, homo_b = mol.get_HOMO_index("a"), mol.get_HOMO_index("b")
    lumo_a, lumo_b = mol.get_LUMO_index("a"), mol.get_LUMO_index("b")
    assert np.abs(mol.occs_a[homo_a] - 1) < 1e-5
    assert np.abs(mol.occs_a[homo_a + 1] - 0) < 1e-5

    assert np.abs(mol.occs_a[lumo_a - 1] - 1) < 1e-5
    assert np.abs(mol.occs_a[lumo_a] - 0) < 1e-5

