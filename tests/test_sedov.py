# tests/test_sedov2d.py

import os
import numpy as np

import diffhydro as dh
from diffhydro.prob_gen import sedov_2d, sedov, make_gaussian_blob
from diffhydro.utils.diagnostics import isotropy_score

def test_sedov_singepoint_strangsplit():
    print("############# SEDOV SYMMETRY TEST #################")
    print("Should take about 1 minute on GPU")

    eq = dh.equationmanager.EquationManager()
    eq.box_size = (1.0, 1.0, 1.0)
    ss = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq,signal_speed=ss)
    cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="SUPERBEE"))
    hydrosim = dh.hydro(n_super_step=200,fluxes=[cf],use_mol=False)
    
    U, _ = sedov_2d(1e7, 0.1, eq)
    
    params = {}
    q = hydrosim.evolve(U,params)
    
    iso = isotropy_score(np.asarray(q[0][0][:, 50, :]))
    assert iso < 0.15, f"Solution not sufficiently isotropic: std/mean={iso:.3f}"


def test_shapes_MOL():
    print("############# SEDOV SYMMETRY TEST - MOL - 3D #################")
    print("Should take about ~1 minute on GPU")

    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = [100,100,100]
    eq.cfl=0.1
    eq.box_size = (1.0, 1.0, 1.0)
    ss = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq,signal_speed=ss)
    cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="SUPERBEE"))
    hydrosim = dh.hydro(n_super_step=100,fluxes=[cf],use_mol=True)
    
    U, _ = make_gaussian_blob(eq, eq.mesh_shape)

    #U, _ = sedov(1e7, 0.1, eq) #3d version
    
    params = {}
    q = hydrosim.evolve(U,params)
    
    iso = isotropy_score(np.asarray(q[0][0][:, 50, :]))
    assert iso < 0.15, f"Solution not sufficiently isotropic: std/mean={iso:.3f}"
