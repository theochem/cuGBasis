import cugbasis
import re
import os
import subprocess
import gbasis
from chemtools.wrappers import Molecule
from iodata import load_one
from gbasis.evals.density import evaluate_density
from gbasis.wrappers import from_iodata
import time
import numpy as np
from grid.cubic import Tensor1DGrids, UniformGrid
import sys

tmpdir = os.environ['SLURM_TMPDIR']
WFN_PATH = [
    #f"{tmpdir}/fchk/atom_08_O_N09_M2_ub3lyp_ccpvtz_g09.wfn",
    #f"{tmpdir}/fchk/ALA_ALA_0_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/qm9_000092_HF_cc-pVDZ.wfn",
    #f"{tmpdir}/fchk/qm9_000104_PBE1PBE_pcS-3.wfn",
    f"{tmpdir}/fchk/PHE_TRP_0_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/YGGFL_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/KLVFF_q001_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/DASXIE_0_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/ZOHMIS_0_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/HHHHHH_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/FRWWHR_q002_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/FEFEFKEK_q-01_m01_k00_force_uwb97xd_def2svpd.wfn",
    #f"{tmpdir}/fchk/DUTLAF10_0_q000_m01_k00_force_uwb97xd_def2svpd.wfn",
]

GRID_SIZES = [2, 10, 25, 50, 75, 100]

np.random.seed(42)
for wfn in WFN_PATH:
    print("FCHK ", wfn)
    # Load Different Molecule Objects
    mol_cc = cugbasis.Molecule(wfn)
    mol_iodata = load_one(wfn)
    mol_c = Molecule.from_file(wfn)

    basis = from_iodata(mol_iodata)
    rdm = (mol_iodata.mo.coeffs * mol_iodata.mo.occs).dot(mol_iodata.mo.coeffs.T) 
    
    # Grid stuff
    min_c, max_c = np.min(mol_iodata.atcoords, axis=0), np.max(mol_iodata.atcoords, axis=0)
    min_c -= 1e-6
    max_c += 1e-6
    print(f"Maximum {max_c}, Minimum {min_c}")
    
    times_of_eval = {
    "grid_sizes": GRID_SIZES,
    "gbasis": [],
    "horton": [],
    "multiwfn": [],
    "cugbasis_wdump": [],
    "cugbasis": [],
    "gpuam": [],
    }
    
    # Create molecular grid for integration
    from grid import MolGrid
    from grid.hirshfeld import HirshfeldWeights
    molgrid2 = MolGrid.from_preset(mol_cc.atnums.astype(int), mol_cc.atcoords, "fine", aim_weights=HirshfeldWeights())

    [times_of_eval[x].append([]) for x in times_of_eval if x != "grid_sizes"]
    for i_numb, numb in enumerate(GRID_SIZES):    
        # Generate an random grid of points around the molecule
        print(numb)
        spacing = np.divide((max_c - min_c + 2.0 * 5.0), numb)
        print(spacing)
        cube_path = f"./cube/cube_{wfn.split('/')[-1].replace('fchk', 'wfn').replace('.wfx', '.wfn')}_{numb}.cube"
        print(cube_path)
        if os.path.exists(cube_path):
            cubicgrid2 = UniformGrid.from_cube(cube_path, weight="Rectangle")
        else:
            axes = np.diag(spacing)
            origin = min_c
            shape = np.array([numb, numb, numb])
            cubicgrid2 = UniformGrid(origin, axes, shape)
            assert cubicgrid2.shape[0] == numb
            cubicgrid2.generate_cube(f"./cube/cube_{wfn.split('/')[-1]}_{cubicgrid2.shape[0]}.cube", np.ones(cubicgrid2.size), mol_iodata.atcoords, mol_iodata.atnums)
        
        print(cubicgrid2.shape, cubicgrid2.size)
        print(cubicgrid2.origin, cubicgrid2.points[-1])

        ngrid = cubicgrid2.points
    
        start = time.time()
        dens = mol_cc.compute_density(ngrid)     
        cubicgrid2.generate_cube(f"{tmpdir}/delete2.cube", dens, mol_iodata.atcoords, mol_iodata.atnums)
        final = time.time()
        execution_time = final - start
        times_of_eval["cugbasis_wdump"][-1].append(execution_time)
        print(f"ChemtoolsCUDA wdump: Execution Time: {execution_time:.6f} seconds")

        start = time.time()
        dens_cc = mol_cc.compute_density(ngrid)     
        final = time.time()
        execution_time = final - start
        times_of_eval["cugbasis"][-1].append(execution_time)
        print(f"ChemtoolsCUDA: Execution Time: {execution_time:.6f} seconds")
        
        sys.stdout.flush()
        
        start = time.time()
        dens_horton = mol_c.compute_density(ngrid)     
        final = time.time()
        execution_time = final - start
        times_of_eval["horton"][-1].append(execution_time)
        print(f"horton: Execution Time: {execution_time:.6f} seconds")
        
         
        start = time.time()
        dens = evaluate_density(rdm, basis, ngrid)
        final = time.time()
        execution_time = final - start
        times_of_eval["gbasis"][-1].append(execution_time)
        print(f"gbasis: Execution Time: {execution_time:.6f} seconds")
        
        print(molgrid2.integrate(mol_cc.compute_density(molgrid2.points)))

        err = np.abs(dens_cc - dens_horton)
        print(f"Chemtools with CC: Maximum err {np.max(err)}")
        print(f"GBASIS with CC: Maximum err {np.max(np.abs(dens - dens_cc))}")
        print(f"GBASIS with Chemtools: Maximum err {np.max(np.abs(dens - dens_horton))}")
        
        # Check with Multiwfn
        try:
            # Run Multiwfn command and capture its output using heredoc
            command = f"time Multiwfn {wfn} -set ./settings.ini -silent <<EOF\n5\n1\n8\n{tmpdir}/{cube_path}\n2\nEOF"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate()
            #print(output)
            print(error)
            match = re.search(r'user\s+(\d+m(\d+\.\d+)s)', error)
            minutes, seconds = match.groups()
            minutes = minutes[:minutes.index("m")]
            total_seconds = float(minutes) * 60 + float(seconds)

            match = re.search(r"wall clock time\s+(\d+)s", output)
            seconds_value = float(match.group(1))
            
            print("Multiwfn total seconds ", seconds_value)
            times_of_eval["multiwfn"][-1].append(seconds_value)
        except Exception as e:
            print(f"Error: {e}")

        # Compare answers with Multiwfn
        process = subprocess.Popen("mv ./density.cub ./density.cube", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
        process.communicate()
        grid, data = UniformGrid.from_cube("./density.cube", return_data=True)
        data = data["data"]
        err = np.abs(data - dens_cc)
        print(f"MULTIWFN vs CC Max error {np.max(err)}")
        print(f"MULTIWFN vs Chemtools Max error {np.max(np.abs(data - dens_horton))}")
        sys.stdout.flush()

        # Run GPUam command and capture its output using heredoc
        o_x, o_y, o_z = cubicgrid2.origin 
        #print(o_x, o_y, o_z)
        nx, ny, nz = cubicgrid2.shape
        print(f"Cubic grid axes {cubicgrid2.axes}")
        sx, sy, sz = np.diag(cubicgrid2.axes)
        print(nx, ny, nz, sx, sy,sz)
        command = f"time ./Gpuam_GPU.x <<EOF\n{wfn}\n{os.environ['SLURM_TMPDIR']}/delete.cube\nB\nE\n{o_x} {o_y} {o_z}\n{nx} {ny} {nz}\n{sx}\nA\nEOF"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        print(output)
        print(error)
        match = re.search("Elapsed time : (\d+\.\d+) s", output)
        seconds = float(match.group(1))
        print("GPUam: Total seconds", seconds)
        
        times_of_eval["gpuam"][-1].append(seconds)

        # Compare GPUAM with GPU
        grid, data = UniformGrid.from_cube(f"{os.environ['SLURM_TMPDIR']}/delete.cube.cube", return_data=True)
        data = data["data"]
        err = np.abs(mol_cc.compute_density(grid.points) - data)
        print(f"GPUAM vs CC (on cube pts) Max error {np.max(err)}")
        print(f"GPUAM vs Chemtools Max Error {np.max(np.abs(data - mol_c.compute_density(grid.points)))}")
        
        print("\n\n\n\n\n\n")
    
    print("\n\n")
    
