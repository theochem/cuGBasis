import numpy as np
import cugbasis
from chemtools.wrappers import Molecule
import pytest
from iodata import load_one
from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density_gradient

mol = cugbasis.Molecule("./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk")
grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
grid_pts = np.array(grid_pts, dtype=np.float64, order="C")
iodata = load_one("./tests/data/atom_01_H_N01_M2_ub3lyp_ccpvtz_g09.fchk")
basis = from_iodata(iodata)
rdm = (iodata.mo.coeffs * iodata.mo.occs).dot(iodata.mo.coeffs.T)
gpu_gradient = mol.compute_gradient(grid_pts)
cpu_gradient = evaluate_density_gradient(rdm, basis, grid_pts)
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
def test_gradient_against_horton(fchk):
    mol = cugbasis.Molecule(fchk)

    grid_pts = np.random.uniform(-5, 5, size=(50000, 3))
    grid_pts = np.array(grid_pts, dtype=np.float64, order="C")

    mol2 = Molecule.from_file(fchk)
    cpu_gradient = mol2.compute_gradient(grid_pts)
    gpu_gradient = mol.compute_gradient(grid_pts)
    assert np.all(np.abs(cpu_gradient - gpu_gradient) < 1e-8)
