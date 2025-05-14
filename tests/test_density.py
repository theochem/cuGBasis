import numpy as np
import cugbasis
from iodata import load_one
from chemtools.wrappers import Molecule
from gbasis.evals.density import evaluate_density
from gbasis.wrappers import from_iodata
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights
import pytest
from test_utils import FCHK_FILES_FOR_TESTING


@pytest.mark.parametrize("fchk", [
    "./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk",
    "./tests/data/atom_he.fchk",
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
    "./tests/data/h2o.fchk",
    "./tests/data/ch4.fchk",
    "./tests/data/qm9_000092_HF_cc-pVDZ.fchk",
    "./tests/data/qm9_000104_PBE1PBE_pcS-3.fchk"
])
def test_electron_density_integral(fchk):
    mol = cugbasis.Molecule(fchk)
    mol_iodata = load_one(fchk)
    mol_grid = MolGrid.from_preset(
        atnums=mol_iodata.atnums,
        atcoords=mol_iodata.atcoords,
        preset="fine",
        aim_weights=BeckeWeights(),
        store=True,
    )
    gpu_density = mol.compute_density(mol_grid.points)
    mol2 = Molecule.from_file(fchk)
    cpu_density = mol2.compute_density(mol_grid.points)
    print("Integral ", mol_grid.integrate(gpu_density))
    print("CPU ", mol_grid.integrate(cpu_density))
    print("Answer ", np.sum(mol_iodata.atnums))
    assert np.abs(np.sum(mol_iodata.atnums) - mol_grid.integrate(gpu_density)) < 1e-3


@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_electron_density_against_horton(fchk):
    spin = np.random.choice(["ab", "a", "b"], size=1)[0]
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_density = mol.compute_density(grid_pts, spin)

    mol2 = Molecule.from_file(fchk)
    cpu_density = mol2.compute_density(grid_pts, spin)

    assert np.all(np.abs(cpu_density - gpu_density) < 1e-8)


@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_electron_density_against_gbasis(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_density = mol.compute_density(grid_pts)

    iodata = load_one(fchk)
    basis = from_iodata(iodata)
    rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)

    cpu_density = evaluate_density(rdm, basis, grid_pts)
    assert np.all(np.abs(cpu_density - gpu_density) < 1e-8)
