"""
RAMSES-RT point-source test, with units that match what the solver actually does.

KEY POINT
---------
The solver is hard-wired to dx = 1 (see hydro.dx_o = 1.0 in hydro_core.py and
ConvectiveFlux_Radiative_transfer.dx_o = 1 in fluxes.py).  Both the flux
divergence (`sol - rhs*dt/self.dx_o`) and the CFL condition
(`dt = cfl / (ndim * c / dx_o)`) use that value.  So the code's length unit is
ONE CELL, not the box.  Therefore the code unit system must be built with

    unit_length = box_width / N          (cell size)

and NOT unit_length = box_width.  With c_code = 1 the radiation front then
advances exactly 1 cell per unit of code time (verified numerically).
"""

import os, sys
sys.path.append("../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

import jax
import jax.numpy as jnp
import numpy as np
import copy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import diffhydro as dh
from diffhydro.units import CodeUnits
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer import StellarRadiationForce

print("Backend:", jax.default_backend(), jax.devices())

# ============================================================================
# PHYSICAL SETUP
# ============================================================================
BASE_OUTPUT_DIR = "examples/athena/Images_athena"


size_shape     = int(os.environ.get("N", 256))
box_width_phys = 4                         # cm
dx_phys        = box_width_phys / size_shape  # cm per cell
source_rate_phys = 1e10                       # photons / s

# ct must stay inside the (periodic) box: ct < box/2 = 0.5 cm  ->  t < 1.67e-11 s.
# The original t = 5.2e-11 s gives ct = 1.56 cm = 1.56 box widths, i.e. the front
# has already wrapped around the periodic domain several times.
t_phys = float(os.environ.get("TPHYS", 5.2e-11))   # s

# --- code units: length unit = ONE CELL ---------------------------------
cu = CodeUnits.from_config(
    {"length": f"{dx_phys} cm", "mass": "1 g", "velocity": "3e10 cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

c_cgs             = 2.99792458e10
light_speed_code  = c_cgs / cu.V_cgs          # ~1.0
dx_cell_code      = 1.0                       # the solver assumes this
time_code         = t_phys / cu.T_cgs
source_rate_code  = source_rate_phys * cu.T_cgs   # photons per code time

print("=" * 70)
print(f"  L_cgs                 = {cu.L_cgs:.6e} cm  (= 1 cell)")
print(f"  V_cgs                 = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs                 = {cu.T_cgs:.6e} s")
print(f"  light_speed_code      = {light_speed_code:.6f}")
print(f"  dx_code               = {dx_cell_code:.6f}")
print(f"  time_code             = {time_code:.4f}")
print(f"  source_rate_code      = {source_rate_code:.4e} photons / code time")
print(f"  expected front  c*t   = {c_cgs * t_phys:.4e} cm "
      f"= {c_cgs * t_phys / dx_phys:.1f} cells "
      f"= {c_cgs * t_phys / box_width_phys:.3f} box widths")
print("=" * 70)

# ============================================================================
# SOLVER
# ============================================================================

eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    debug=False,
)
solver_test = dh.HLL_Radiative_transfer_Local(
    equation_manager=eq_test, signal_speed=dh.signal_speed_Rusanov
)
# NOTE: ConvectiveFlux_Radiative_transfer takes no `dx` argument -- it uses dx_o = 1.
cf_test = dh.ConvectiveFlux_Radiative_transfer(eq_test, solver_test, dh.PLM(limiter="VANLEER"))

stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=dx_cell_code,               # = 1: only used as cell volume, unused in "stromgren"
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=False,
    gaussian_star=True,            # gaussian_star=False breaks under jit (python `if` on tracer)
    injection_geometry="3D",
    eq=eq_test,
    debug=False,
    momentum_only=False,
)

hydrosim_test = dh.hydro(n_super_step=10000, fluxes=[cf_test], forces=[stellar_force])
print("hydrosim_test.dx_o =", hydrosim_test.dx_o, " cf.dx_o =", cf_test.dx_o,
      " cfl =", eq_test.cfl)
print("expected dt_code   =", eq_test.cfl / (3.0 * light_speed_code / 1.0))

params = {
    "star_masses":        jnp.array([1.0]),
    "star_ages":          jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions":     jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}
sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

print(f"\nRunning to t = {t_phys:.3e} s = {time_code:.2f} code units ...")
field_test, _, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test), params, time_code
)

dt_hist = np.asarray(dt_hist)
dt_sum  = float(dt_hist[dt_hist > 0].sum())
print("Done.")
print(f"  steps           = {n_steps},  dt_code = {dt_hist[0]:.6e}")
print(f"  sum(dt)         = {dt_sum:.4f} code = {dt_sum * cu.T_cgs:.4e} s")

# ============================================================================
# DIAGNOSTICS
# ============================================================================

E3d = np.asarray(field_test[0])
c   = size_shape // 2
print(f"  E min/max       = {E3d.min():.4e} / {E3d.max():.4e}")
print(f"  photons in box  = {E3d.sum():.6e}   expected = {source_rate_code * dt_sum:.6e}")

line = E3d[c:, c, c]
peak = E3d.max()
for th in [1e-3, 1e-6, 1e-10, 1e-15]:
    idx = np.where(line > peak * th)[0]
    r = idx.max() if idx.size else 0
    print(f"  thr {th:.0e} of peak -> radius {r:3d} cells = {r * dx_phys:.4e} cm")
print(f"  expected free-streaming radius = {c_cgs * t_phys / dx_phys:.1f} cells")

# ============================================================================
# PLOT
# ============================================================================

plt.style.use("dark_background")
E_slice = E3d[:, :, c]
extent  = [-box_width_phys / 2, box_width_phys / 2] * 2

fig, ax = plt.subplots(figsize=(6, 5))
pos = E_slice[E_slice > 0]
im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot",
               extent=extent, norm=LogNorm(vmin=max(pos.min(), peak * 1e-12), vmax=peak))
ax.set_xlabel("y [cm]"); ax.set_ylabel("x [cm]")
ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per cell")
plt.tight_layout()
out = f"{BASE_OUTPUT_DIR}/field_test_fixed_units.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
