import numpy as np
import cugbasis
from iodata import load_one
from grid.cubic import UniformGrid
from chemtools.wrappers import Molecule
from gbasis.evals.density import evaluate_density
from gbasis.wrappers import from_iodata
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights
import pytest
from test_utils import FCHK_FILES_FOR_TESTING


@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_electron_density_integral(fchk, accuracy):
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
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    gpu_density = mol.compute_density(grid_pts)

    mol2 = Molecule.from_file(fchk)
    cpu_density = mol2.compute_density(grid_pts)

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
