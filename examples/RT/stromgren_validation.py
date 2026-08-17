"""
Physical validation of the hydrogen chemistry: R-type expansion of an HII
region in a uniform static medium (Iliev et al. 2006, "Test 1").

Analytic reference (isothermal, case B, pure hydrogen, static gas):

    R_I(t) = R_S [1 - exp(-t / t_rec)]^(1/3)

    R_S    = [3 Q / (4 pi alpha_B n_H^2)]^(1/3)      Stromgren radius
    t_rec  = 1 / (alpha_B n_H)                        recombination time

The run uses the reduced speed of light approximation (RSLA) so the
recombination time is reachable with an explicit solver, and disables
hydrodynamic feedback (no heating -> the gas stays isothermal, which is the
assumption of the reference solution).

    python stromgren_validation.py            # default N = 32
    N=48 python stromgren_validation.py

It prints R_I(t)/R_S against the analytic curve and writes a PNG.
"""

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))

import jax
# float64 is MANDATORY here: sol[0] is a photon density per CODE volume, so it
# scales as L_cgs^3. With a kpc-sized length unit that is ~1e59 photons per
# code volume, which overflows float32 (max 3.4e38) and produces NaN.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import (
    EquationManager as EquationManagerRT,
)
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits

print("Backend:", jax.default_backend())

# ---------------------------------------------------------------------------
# Iliev+2006 Test 1 parameters
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", 150))
n_H_cgs = 1.0e-3                 # cm^-3
T_K = 1.0e4                      # K, isothermal
Q_phot = 5.0e48                  # ionizing photons / s
GAMMA = 5.0 / 3.0

alpha_B = float(hchem.alpha_B_HII_cgs(T_K))
alpha_A = float(hchem.alpha_A_HII_cgs(T_K))
R_S = (3.0 * Q_phot / (4.0 * np.pi * alpha_B * n_H_cgs ** 2)) ** (1.0 / 3.0)
t_rec = 1.0 / (alpha_B * n_H_cgs)

box_cgs = 4.0 * R_S / 1.4        # box comfortably larger than 2 R_S
dx_cgs = box_cgs / N
t_end = float(os.environ.get("TEND", 5.0)) * t_rec    # in units of t_rec
print(t_end)
# RSLA: c_red must stay above the early ionization-front speed.  The old
# 1e-3 default was slower than the front at one cell for this setup, which
# adds an artificial early-time delay before any chemistry issue is involved.
rsla = float(os.environ.get("RSLA", 2.0e-2))
c_red_cgs = rsla * hchem.C_LIGHT_CGS
v_front_typ = Q_phot / (4.0 * np.pi * (0.5 * R_S) ** 2 * n_H_cgs)

# ---------------------------------------------------------------------------
# Code units: 1 length unit = 1 cell, 1 velocity unit = c_red
# ---------------------------------------------------------------------------
# mass unit chosen so that rho_code = 1 for the ambient medium: with
# "mass = 1 g" and a kpc length unit, rho_code would be ~1e37 and every
# derived quantity would sit near the float32 ceiling.
M_unit_cgs = n_H_cgs * hchem.MH_CGS * dx_cgs ** 3
cu = CodeUnits.from_config(
    {"length": f"{dx_cgs} cm", "mass": f"{M_unit_cgs} g",
     "velocity": f"{c_red_cgs} cm/s"},
    {"gamma": GAMMA, "mu": 1.0},
)
dx_code = 1.0
c_code = c_red_cgs / cu.V_cgs          # == 1
cfl = 0.4
dt_code = cfl / (3.0 * c_code / dx_code)
n_steps = int(math.ceil((t_end / cu.T_cgs) / dt_code))
v_front_cell = Q_phot / (4.0 * np.pi * dx_cgs ** 2 * n_H_cgs)

print("=" * 72)
print(f"  N              = {N},  dx = {dx_cgs:.4e} cm = {dx_cgs / 3.0857e18:.3f} pc")
print(f"  box            = {box_cgs:.4e} cm = {box_cgs / 3.0857e18:.2f} pc")
print(f"  n_H            = {n_H_cgs:.3e} cm^-3,  T = {T_K:.1e} K,  Q = {Q_phot:.2e} ph/s")
print(f"  alpha_B(1e4 K) = {alpha_B:.4e} cm^3/s")
print(f"  R_S            = {R_S:.4e} cm = {R_S / 3.0857e18:.2f} pc = {R_S / dx_cgs:.2f} cells")
print(f"  t_rec          = {t_rec:.4e} s = {t_rec / 3.156e13:.1f} Myr")
print(f"  tau per cell   = {n_H_cgs * hchem.SIGMA_HI_0_CGS * dx_cgs:.3e}")
print(f"  RSLA           = {rsla:.1e}  ->  c_red = {c_red_cgs:.3e} cm/s")
print(f"                   v_front(R_S/2) = {v_front_typ:.3e} cm/s "
      f"(margin x{c_red_cgs / v_front_typ:.1f})")
print(f"                   v_front(dx) = {v_front_cell:.3e} cm/s "
      f"(margin x{c_red_cgs / v_front_cell:.1f})")
if c_red_cgs < v_front_cell:
    print("  !! WARNING: RSLA is slower than the early ionization front; "
          "the analytic early-time curve cannot be recovered.")
print(f"  dt             = {dt_code * cu.T_cgs:.3e} s,  n_steps = {n_steps}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
eq_rt = EquationManagerRT(
    light_speed=c_code, mesh_shape=(N, N, N), eps=1e-30, debug=False,
)
eq_hydro = dh.EquationManager(
    gamma=GAMMA, n_cons=6, passive_names=("x_HII",),
    mesh_shape=(N, N, N), eps=1e-30,
)


class StateBlockFlux:
    def __init__(self, base_flux, state_slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o

    def flux(self, sol, ax, params, flux):
        local = self.base_flux.flux(sol[self.state_slice], ax, params, flux)
        return jnp.zeros_like(sol).at[self.state_slice].set(local)

    def timestep(self, sol):
        return self.base_flux.timestep(sol[self.state_slice])


rt_flux = StateBlockFlux(
    dh.ConvectiveFlux_Radiative_transfer(
        eq_rt, dh.LaxFriedrichs_Radiative_transfer(eq_rt, dh.signal_speed_Rusanov),
        dh.PLM(limiter="VANLEER"), dx=dx_code),
    slice(0, eq_rt.n_cons),
)
hydro_flux = StateBlockFlux(
    dh.ConvectiveFlux(
        eq_hydro, dh.LaxFriedrichs(eq_hydro, dh.signal_speed_Rusanov),
        dh.PLM(limiter="VANLEER"), dx=dx_code),
    slice(eq_rt.n_cons, eq_rt.n_cons + eq_hydro.n_cons),
)

stellar = StellarRadiationForce(
    dx=dx_code,
    injection_mode="stromgren",
    stromgren_rate=Q_phot * cu.T_cgs,       # photons per code time
    injection_momentum=False,
    injection_geometry="radial_3D",         # isotropic star, not a +x beam
    gaussian_star=True,
    beam_momentum_scaling="legacy_c2_source2",
    eq=eq_rt, hydro_eq=eq_hydro, cu=cu,
    chemistry=False,                       # injection only
)
chem_force = HydrogenPhotoChemistryForce(   
    stellar,
    case="B",
    collisional=False,
    max_frac=0.9,
    # Iliev Test 1 is isothermal.  The coupled force is still responsible
    # for the energy slot, but projects it so T stays exactly 10^4 K after
    # x_HII changes; H-L is deliberately disabled for this analytic test.
    include_heating=False,
    include_cooling=False,
    # fixed_temperature_K=T_K,
)
print(f"  chemistry scheme = coupled ({type(chem_force).__name__})")

sim = dh.hydro(
    n_super_step=n_steps + 10,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar, chem_force],
    dx=dx_code,
    max_dt=2.0 * dt_code,
)

# ---------------------------------------------------------------------------
# Initial conditions (CONSERVATIVE state)
# ---------------------------------------------------------------------------
rho_code = n_H_cgs * hchem.MH_CGS / cu.rho_cgs
p_code = n_H_cgs * hchem.KB_CGS * T_K / cu.P_cgs        # x_HII = 0 -> n_tot = n_H
sol = jnp.zeros((10, N, N, N), dtype=jnp.float64)
sol = sol.at[4].set(rho_code)
sol = sol.at[8].set(p_code / (GAMMA - 1.0))
c_idx = N // 2

params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.0]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[c_idx] * 3], dtype=jnp.int32),
}

# ---------------------------------------------------------------------------
# Time loop with snapshots
# ---------------------------------------------------------------------------
n_snap = 12
snap_every = max(1, n_steps // n_snap)


def ionized_radius(x3d):
    """Volume-equivalent radius of the ionized region."""
    V_ion = float(np.sum(np.asarray(x3d, dtype=np.float64))) * dx_cgs ** 3
    return (3.0 * V_ion / (4.0 * np.pi)) ** (1.0 / 3.0)


def xHII_from_conservative(sol):
    """Recover x_HII from the conserved rho*x_HII field for diagnostics."""
    return np.asarray(chem_force.view.xHII(sol), dtype=np.float64)


@jax.jit
def run_chunk(sol, params, k0):
    """`snap_every` fixed-dt steps, compiled once."""
    def body(j, carry):
        s, p = carry
        return sim._hydrostep(k0 + j, (s, p), dt_code)
    return jax.lax.fori_loop(0, snap_every, body, (sol, params))


times, radii = [0.0], [0.0]
t_code = 0.0
n_chunks = max(1, n_steps // snap_every)
for chunk in range(n_chunks):
    sol, params = run_chunk(sol, params, chunk * snap_every)
    t_code += snap_every * dt_code
    k = (chunk + 1) * snap_every - 1
    if True:
        x3d = xHII_from_conservative(sol)
        t_s = t_code * cu.T_cgs
        times.append(t_s)
        radii.append(ionized_radius(x3d))
        print(f"  step {k+1:5d}/{n_steps}  t/t_rec = {t_s / t_rec:6.3f}  "
              f"R_I/R_S = {radii[-1] / R_S:7.4f}  "
              f"(analytic {(1 - math.exp(-t_s / t_rec)) ** (1/3):7.4f})  "
              f"x_max = {x3d.max():.4f}")

times = np.array(times)
radii = np.array(radii)
analytic = R_S * (1.0 - np.exp(-times / t_rec)) ** (1.0 / 3.0)

rel_err = np.abs(radii[1:] - analytic[1:]) / analytic[1:]
print("=" * 72)
print(f"  final  R_I = {radii[-1]:.4e} cm,  analytic = {analytic[-1]:.4e} cm,  "
      f"ratio = {radii[-1] / analytic[-1]:.4f}")
print(f"  mean |relative error| over the run = {rel_err.mean() * 100:.2f} %")
print(f"  (a few % is expected: R_S spans only {R_S / dx_cgs:.1f} cells)")
print("=" * 72)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
out_dir = os.path.join(REPO_ROOT, "examples/RT/Images/stromgren_validation")
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
t_fine = np.linspace(0, times[-1], 400)
axes[0].plot(t_fine / t_rec, (1 - np.exp(-t_fine / t_rec)) ** (1 / 3),
             "k-", lw=2, label=r"analytic $[1-e^{-t/t_{rec}}]^{1/3}$")
axes[0].plot(times / t_rec, radii / R_S, "o--", color="tab:red", ms=5,
             label=f"DiffHydro (N={N})")
axes[0].set_xlabel(r"$t / t_{\rm rec}$")
axes[0].set_ylabel(r"$R_I / R_S$")
axes[0].set_title("R-type expansion of the HII region")
axes[0].legend()
axes[0].grid(alpha=0.3)

x_slice = xHII_from_conservative(sol)[:, :, c_idx]
extent = [-0.5 * box_cgs / 3.0857e18, 0.5 * box_cgs / 3.0857e18] * 2
im = axes[1].imshow(x_slice.T, origin="lower", cmap="magma", vmin=0, vmax=1,
                    extent=extent)
th = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(R_S / 3.0857e18 * np.cos(th), R_S / 3.0857e18 * np.sin(th),
             "c--", lw=1.5, label=r"$R_S$")
axes[1].set_xlabel("x [pc]"); axes[1].set_ylabel("y [pc]")
axes[1].set_title(rf"$x_{{\rm HII}}$ at $t = {times[-1]/t_rec:.1f}\,t_{{\rm rec}}$")
axes[1].legend()
fig.colorbar(im, ax=axes[1], label=r"$x_{\rm HII}$")
plt.tight_layout()
out = os.path.join(out_dir, f"stromgren_N{N}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
