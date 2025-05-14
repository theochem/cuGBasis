import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest
from iodata import load_one
from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density_gradient
from test_utils import FCHK_FILES_FOR_TESTING

@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_gradient_against_gbasis(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
    iodata = load_one(fchk)
    basis = from_iodata(iodata)
    rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
    gpu_gradient = mol.compute_gradient(grid_pts)
    cpu_gradient = evaluate_density_gradient(rdm, basis, grid_pts)

    assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)


@pytest.mark.parametrize("fchk", FCHK_FILES_FOR_TESTING)
def test_gradient_against_horton(fchk):
    spin = np.random.choice(["ab", "a", "b"], size=1)[0]

    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")

    mol2 = Molecule.from_file(fchk)
    cpu_gradient = mol2.compute_gradient(grid_pts, spin)
    gpu_gradient = mol.compute_gradient(grid_pts, spin)
    assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)
