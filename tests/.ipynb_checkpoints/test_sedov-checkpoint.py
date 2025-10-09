# tests/test_sedov2d.py

import os

#using GPU, comment out if on CPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np

import jax.numpy as jnp
import diffhydro as dh

def sedov_2d(E0,rho):
        #doesn't really work yet for non-square boxes...
        dV = (100)**2
        hc = int(100/2)-1
        sol = jnp.zeros((5,100,100,100))
        sol = sol.at[0].set(rho)
        sol = sol.at[-1].set(1.0E-3)
        sol = sol.at[4,hc,hc,hc].set(E0/dV)
        rmax = 3.0*(100/2.0)/4.0
        tf = jnp.sqrt((rmax/1.15)**5/E0)
        return sol, tf

def isotropy_score(rho, xlen, ylen, nbins=36):
    nx, ny = rho.shape
    dx, dy = xlen/nx, ylen/ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    Xc, Yc = X - 0.5*xlen, Y - 0.5*ylen
    r = np.sqrt(Xc**2 + Yc**2)
    theta = (np.arctan2(Yc, Xc) + 2*np.pi) % (2*np.pi)

    rho_np = np.asarray(rho)
    rho0 = np.median(rho_np)
    mask = rho_np >= 1.5 * rho0
    if not np.any(mask):
        return np.inf

    # angular bins: take 95th percentile radius per bin where mask is true
    edges = np.linspace(0, 2*np.pi, nbins+1)
    radii = []
    for i in range(nbins):
        sel = (theta >= edges[i]) & (theta < edges[i+1]) & mask
        if np.any(sel):
            radii.append(np.percentile(r[sel], 95))
    if len(radii) < 3:
        return np.inf
    radii = np.array(radii)
    return float(np.std(radii) / (np.mean(radii) + 1e-12))
    
def test_shapes():
    print("############# SEDOV SYMMETRY TEST #################")
    print("Should take about 1 minute on GPU")

    eq = dh.equationmanager.EquationManager()
    ss = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq,signal_speed=ss)
    cf = dh.ConvectiveFlux(eq,solver,dh.MUSCL3(limiter="SUPERBEE"))
    hydrosim = dh.hydro(n_super_step=300,fluxes=[cf],maxjit=True)
    
    U,_ = sedov_2d(1E7,0.1)
    
    params = {}
    q = hydrosim.evolve(U,params)
    
    iso = isotropy_score(q[0][0][:,50,:],1,1)
    assert iso < 0.15, f"Solution not sufficiently isotropic: std/mean={iso:.3f}"
