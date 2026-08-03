
"""
run_coupled_rhd_example.py

Minimal end-to-end driver: builds the combined hydro+RT+chem state described
in coupled_rhd.py, runs the simulation to a target physical time, and shows
how to pull out EVERY field (rho, vx,vy,vz,p, E_gamma,Fx,Fy,Fz, x_HII) from
the single returned tensor at the end.

Adapt the object construction (equation managers, ConvectiveFlux, forces) to
match your ACTUAL constructors -- the names/kwargs below mirror what you
showed in your Athena/RAMSES-RT test script and cooling-15.py.
"""

import numpy as np
import jax.numpy as jnp
import diffhydro as dh                                   # your package
from diffhydro.coupled_rhd import (
    EquationManagerCoupled, BlockFlux, ChemBlockFlux,
    IonizationForce, ChemistryBoundsForce, build_coupled_hydro_example,
)
import os


# ----------------------------------------------------------------------------
# 1) Build the two INDEPENDENT equation managers exactly as you already do
# ----------------------------------------------------------------------------
size_shape = 64
dx_code = 1.0

hydro_eq = dh.EquationManager(          # ADAPT: your hydro EquationManager class
    gamma=5.0 / 3.0,
    n_cons=5,
)

rt_eq = dh.EquationManager_RT(          # ADAPT: import path used in your script
    light_speed=0.05,                   # REDUCED speed of light (RSLA), not 1.0/c_phys
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=1e-20,
    n_cons=4,                           # RT block itself has NO passive scalar here;
                                          # x_HII lives in the coupled chem_slice instead
)

# ----------------------------------------------------------------------------
# 2) Build each block's OWN ConvectiveFlux (own solver/reconstruction/dx)
# ----------------------------------------------------------------------------
solver_hydro = dh.LaxFriedrichs(equation_manager=hydro_eq, signal_speed=dh.signal_speed_Rusanov)  # ADAPT
cf_hydro = dh.ConvectiveFlux(hydro_eq, solver_hydro, dh.PLM(limiter="VANLEER"), dx=dx_code)         # ADAPT

solver_rt = dh.LaxFriedrichs_Radiative_transfer(equation_manager=rt_eq, signal_speed=dh.signal_speed_Rusanov)
cf_rt = dh.ConvectiveFlux_Radiative_transfer(rt_eq, solver_rt, dh.PLM(limiter="VANLEER"), dx=dx_code)

# ----------------------------------------------------------------------------
# 3) Forces: stellar injection (writes into the RT block) + cooling (hydro
#    block). IonizationForce and ChemistryBoundsForce are added automatically
#    inside build_coupled_hydro_example.
# ----------------------------------------------------------------------------
stellar_force = dh.physics.radiative_transfer.StellarRadiationForce(   # ADAPT import path
    escape_fraction=0.1, dx=dx_code, injection_mode="stromgren",
    stromgren_rate=1e2, gaussian_star=True, injection_geometry="3D",
    eq=rt_eq, debug=False, momentum_only=False,
)

heatcool_force = dh.physics.cooling.HeatCoolForce(                      # ADAPT import path
    equation_manager=hydro_eq, pressure_fn=None,
    logT_table=np.linspace(4.0, 8.5, 91),
    logLambda_m20_table=np.zeros(91),      # ADAPT: your real cooling table
)

# NOTE: stellar_force writes into indices 0..3 (its own local convention);
# since it is called through `hydro.forcing()` on the FULL combined tensor
# U, it must be wrapped so it only touches coupled_eq.rt_slice. Simplest
# fix: give it its own thin adapter (see RTForceAdapter below) rather than
# passing it directly, unless you already generalized StellarRadiationForce
# to accept an arbitrary var_slice.

from dataclasses import dataclass

@dataclass
class SliceForceAdapter:
    """Wrap an existing force so it only sees/writes its own var_slice of
    the combined tensor U, exactly like BlockFlux does for fluxes."""
    inner_force: object
    var_slice: slice
    n_cons_total: int

    def timestep(self, U):
        return self.inner_force.timestep(U[self.var_slice])

    def force(self, i_step, U, params, dt):
        sol_block = U[self.var_slice]
        sol_block_new, params = self.inner_force.force(i_step, sol_block, params, dt)
        U = U.at[self.var_slice].set(sol_block_new)
        return U, params

# ----------------------------------------------------------------------------
# 4) Assemble the coupled system
# ----------------------------------------------------------------------------
coupled_eq = EquationManagerCoupled(hydro_eq=hydro_eq, rt_eq=rt_eq)
n_cons_total = coupled_eq.n_cons   # 5 + 4 + 1 = 10

hydro_flux = BlockFlux(inner_flux=cf_hydro, var_slice=coupled_eq.hydro_slice, n_cons_total=n_cons_total)
rt_flux = BlockFlux(inner_flux=cf_rt, var_slice=coupled_eq.rt_slice, n_cons_total=n_cons_total)
chem_flux = ChemBlockFlux(hydro_eq=hydro_eq, hydro_slice=coupled_eq.hydro_slice,
                           chem_slice=coupled_eq.chem_slice, n_cons_total=n_cons_total)

stellar_force_wrapped = SliceForceAdapter(inner_force=stellar_force,
                                           var_slice=coupled_eq.rt_slice,
                                           n_cons_total=n_cons_total)
heatcool_force_wrapped = SliceForceAdapter(inner_force=heatcool_force,
                                            var_slice=coupled_eq.hydro_slice,
                                            n_cons_total=n_cons_total)
ionization_force = IonizationForce(coupled_eq=coupled_eq)
bounds_force = ChemistryBoundsForce(coupled_eq=coupled_eq)

hydrosim = dh.hydro(
    n_super_step=2000,
    fluxes=[hydro_flux, rt_flux, chem_flux],
    forces=[stellar_force_wrapped, ionization_force, heatcool_force_wrapped, bounds_force],
    dx=dx_code,
    max_dt=0.1,
)

# ----------------------------------------------------------------------------
# 5) Initial condition: full (10, Nx, Ny, Nz) primitive state, then convert
# ----------------------------------------------------------------------------
prim0 = jnp.zeros((n_cons_total, size_shape, size_shape, size_shape))
prim0 = prim0.at[coupled_eq.i_rho].set(1.0)      # uniform background density
prim0 = prim0.at[coupled_eq.i_p].set(1.0)        # uniform background pressure
# velocities, E_gamma, F_gamma, x_HII all start at 0 (already zeros)

sol0 = coupled_eq.get_conservatives_from_primitives(prim0)

params0 = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}

# ----------------------------------------------------------------------------
# 6) RUN: evolve until a target physical/code time
# ----------------------------------------------------------------------------
t_target_code = 5.0   # ADAPT: your physical time converted to code units

field_final, params_final, t_final, dt_hist, n_steps = hydrosim.evolve_till_time(
    sol0, params0, t_target_code
)

# ----------------------------------------------------------------------------
# 7) EXTRACT every field from the single returned tensor `field_final`
# ----------------------------------------------------------------------------
primitives_final = coupled_eq.get_primitives_from_conservatives(field_final)

rho     = np.asarray(primitives_final[coupled_eq.i_rho])
vx, vy, vz = (np.asarray(primitives_final[i]) for i in coupled_eq.i_vel)
p       = np.asarray(primitives_final[coupled_eq.i_p])

E_gamma = np.asarray(primitives_final[coupled_eq.i_Egamma])
Fx, Fy, Fz = (np.asarray(primitives_final[i]) for i in coupled_eq.i_Fgamma)

x_HII   = np.asarray(primitives_final[coupled_eq.i_xHII])

print("t_final (code) =", float(t_final), " n_steps =", int(n_steps))
print("rho     min/max =", rho.min(), rho.max())
print("p       min/max =", p.min(), p.max())
print("E_gamma min/max =", E_gamma.min(), E_gamma.max())
print("x_HII   min/max =", x_HII.min(), x_HII.max())

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

np.savez(
    "output/coupled_rhd_final_state.npz",
    rho=rho, vx=vx, vy=vy, vz=vz, p=p,
    E_gamma=E_gamma, Fx=Fx, Fy=Fy, Fz=Fz, x_HII=x_HII,
    t_final=float(t_final), n_steps=int(n_steps),
)
print("saved -> output/coupled_rhd_final_state.npz")
