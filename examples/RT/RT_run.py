"""
RAMSES-RT point-source test, with units that match what the solver actually does.

UNIT CONVENTION (new, fully unit-agnostic via UnitParser)
----------------------------------------------------------
Every physical length/velocity/time input can now be given in ANY unit known to
diffhydro.units.registry.UnitParser (cm, m, km, pc, kpc for length; cm/s, m/s,
km/s for velocity; s for time; ...). Each input is parsed once, converted to cgs
immediately, and everything downstream (dx_code, CodeUnits, the solver, the
plots) only ever sees cgs values -- so changing the unit of an input is
completely transparent and never requires touching the rest of the script.

    old :   unit_length   = box_width_phys / N      (1 cell = 1 code unit,
                                                        dx_code == 1 enforced)
    new:   box_width_phys_cgs = box_width_code * unit_length_phys_cgs
           dx_code            = box_width_code / N     (arbitrary dx_code)

The two FREE inputs are now
  * unit_length_phys : physical size of ONE code length unit, given as a
                        free string with any supported unit, e.g. "1 km",
                        "0.05 cm", "3.2e-3 pc",
  * box_width_phys   : physical box size, also a free unit string
                        (e.g. "10 km", "3.2 cm"), takes precedence over
                        box_width_code if both are given,
and both the physical box size and the cell size follow from them, always
expressed internally in cgs (cm).

Consequence: dx_code is generally != 1, so it must be passed explicitly
to EVERYTHING that contains a dx, otherwise the solver falls back to its
default dx_o = 1:

  * hydro(dx=dx_code)                        -> flux divergence, rhs/dx_o
  * ConvectiveFlux_Radiative_transfer(dx=)   -> CFL, dt = cfl / (ndim*c/dx)
  * StellarRadiationForce(dx=dx_code)        -> source cell volume

And the sol[0] field (E_gamma) is a photon DENSITY in code units
(photons per code-volume unit), not "photons per cell":

    photons per cell    = E_code * dx_code**3
    n_gamma [cm^-3]      = E_code / cu.L_cgs**3      (= photons_per_cell / dx_phys_cgs**3)
    photons in the box   = sum(E_code) * dx_code**3

All the diagnostics/plotting below work with E_cell (photons per cell), which
is invariant under a change of unit convention.

Environment variables (ULEN, BOXPHYS, UVEL, TPHYS accept ANY unit string
recognized by UnitParser, e.g. "1 km", "3.2 cm", "3e5 km/s", "5.2e-11 s"):
  GPU, N, ULEN, BOXPHYS, BOXCODE, UVEL, SRC, TPHYS, EPS, MAXDT, NSTEP,
  RTRUNC_AVG, RTRUNC_FIT

If BOXPHYS is set, it takes precedence over BOXCODE. SRC (photons/s) has no
length/mass/time dimension in the unit table, so it stays a plain float
(interpreted as cgs photons/s).
"""

import os, sys, math

from diffhydro.physics.fraction_xHII import HydrogenIonizationForce as HydrogenIonizationForce
# repository root, so the script can run from any cwd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

import jax
import jax.numpy as jnp
import numpy as np
import copy as cp
import matplotlib
matplotlib.use("Agg")  # whether or not to display all plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import diffhydro as dh
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.hydrogen_chemistry import MH_CGS as mH_chem
from diffhydro.units import CodeUnits
from diffhydro.units.registry import UnitParser  # <- your unit parser/registry
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.coupled_rhd import (
    EquationManagerCoupled, BlockFlux, ChemBlockFlux,
    IonizationForce, ChemistryBoundsForce, build_coupled_hydro_example,
)
print("Backend:", jax.default_backend(), jax.devices())

up = UnitParser()

def env_quantity(name: str, default: str, expected_dim: str):
    """Read an env var as a free-form quantity string ('1 km', '3.2 cm',
    '3e10 cm/s', '5.2e-11 s', ...), parse it with UnitParser, and return the
    ParsedQuantity (value, unit, dimension, cgs_value). This is the single
    choke point that makes unit choice fully transparent to the rest of the
    script: everything downstream only ever reads `.cgs_value`."""
    text = os.environ.get(name, default)
    try:
        return up.parse(text, expected_dim=expected_dim)
    except ValueError as exc:
        raise SystemExit(f"Invalid value for {name}='{text}': {exc}") from exc

# ============================================================================
# PHYSICAL SETUP  (unit-agnostic: any input unit is converted to cgs here,
# and only cgs values are used from this point on)
# ============================================================================

size_shape = int(os.environ.get("N", 100))

# --- free inputs of the new convention, parsed via UnitParser --------------
ulen_q = env_quantity("ULEN", "1.0 cm", expected_dim="length")
unit_length_phys_cgs = ulen_q.cgs_value        # cm  <- 1 code length unit
unit_length_str      = f"{ulen_q.value:g} {ulen_q.unit}"   # for printing/tagging

uvel_q = env_quantity("UVEL", "3e10 cm/s", expected_dim="velocity")
unit_velocity_phys = uvel_q.cgs_value          # cm/s <- 1 code velocity unit

# Box size: either BOXPHYS (any length unit string) or BOXCODE (dimensionless
# code units). BOXPHYS takes precedence if provided.
if "BOXPHYS" in os.environ:
    boxphys_q = env_quantity("BOXPHYS", "3.2 cm", expected_dim="length")
    box_width_phys_cgs = boxphys_q.cgs_value                 # cm
    box_width_code     = box_width_phys_cgs / unit_length_phys_cgs
    box_width_str      = f"{boxphys_q.value:g} {boxphys_q.unit}"
else:
    box_width_code = float(os.environ.get("BOXCODE", 3.2))  # code length units
    box_width_phys_cgs = box_width_code * unit_length_phys_cgs
    box_width_str = f"{box_width_phys_cgs:.3e} cm"

# --- everything else is derived, all in cgs ---------------------------------
dx_code          = box_width_code / size_shape          # code units per cell
dx_phys_cgs      = dx_code * unit_length_phys_cgs        # cm per cell
cell_volume_code = dx_code ** 3                          # cell volume, code units
cell_volume_cm3  = dx_phys_cgs ** 3                      # cell volume, cm^3

# Source rate: photons/s has no length/mass/time dimension in the unit table,
# so it is kept as a plain float (cgs photons/s by convention).
source_rate_phys = float(os.environ.get("SRC", 1e10))    # photons / s

# ct must stay inside the (periodic) box: ct < box/2 -> t < box/(2c).
# The original t = 5.2e-11 s gives ct = 1.56 cm = 0.49 box widths for box = 3.2 cm.
tphys_q = env_quantity("TPHYS", "5.2e-11 s", expected_dim="time")
t_phys = tphys_q.cgs_value   # s

# --- code units: 1 code length unit = unit_length_phys_cgs cm --------------
cu = CodeUnits.from_config(
    {"length": f"{unit_length_phys_cgs} cm",
     "mass": "1 g",
     "velocity": f"{unit_velocity_phys} cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

c_cgs            = 2.99792458e10
# RSLA: reduced speed of light approximation (Rosdahl+ 2013, sec. 3.3).
# RSLA=1 -> true c. RSLA=1e-3 -> c_red = c/1000, which makes it possible to
# reach recombination timescales (~1e15 s) with an explicit RT solver. The
# chemistry uses the SAME reduced c in the interaction terms, so the
# equilibrium (Stromgren) solution is preserved; only the time to reach it
# is stretched. c_red must stay well above the ionization-front speed.
rsla_factor = float(os.environ.get("RSLA", 1.0))
light_speed_code = rsla_factor * c_cgs / cu.V_cgs   # ~1.0 if UVEL ~ c and RSLA=1
time_code        = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs   # photons per code time

# CFL of the RT flux: dt = cfl / (ndim * c / dx) -> now depends on dx_code.
cfl_code    = 0.4                                  # = EquationManager_RT.cfl
dt_cfl      = cfl_code / (3.0 * light_speed_code / dx_code)
n_steps_est = int(math.ceil(time_code / dt_cfl))
# max_dt must not override the CFL limit (hydro default = 0.5, which caps large dx_code).
max_dt       = float(os.environ.get("MAXDT", 2.0 * dt_cfl))
n_super_step = int(os.environ.get("NSTEP", int(1.2 * n_steps_est) + 100))

# ============================================================================
# RUN TAG: encodes every physical input parameter (values reported in cgs,
# regardless of what unit string the user typed), used for BOTH the output
# folder name and every saved filename.
# ============================================================================
run_tag = (
    f"N{size_shape}"
    f"_ulen{unit_length_phys_cgs:.2e}cm"
    f"_boxc{box_width_code:.2e}"
    f"_box{box_width_phys_cgs:.2e}cm"
    f"_v{cu.V_cgs:.2e}cms"
    f"_src{source_rate_phys:.2e}phs"
    f"_t{t_phys:.2e}s"
)

BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/RT/Images", run_tag)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(f"  run_tag               = {run_tag}")
print(f"  output dir            = {BASE_OUTPUT_DIR}")
print("=" * 70)
print("  --- user-facing inputs (any unit, auto-converted to cgs) ---")
print(f"  ULEN input            = '{unit_length_str}'  -> {unit_length_phys_cgs:.6e} cm")
print(f"  BOX  input            = '{box_width_str}'    -> {box_width_phys_cgs:.6e} cm")
print(f"  UVEL input            = '{uvel_q.value:g} {uvel_q.unit}' -> {unit_velocity_phys:.6e} cm/s")
print(f"  TPHYS input           = '{tphys_q.value:g} {tphys_q.unit}' -> {t_phys:.6e} s")
print("  --- convention box_size = box_width_code * unit_length ---")
print(f"  unit_length_phys_cgs  = {unit_length_phys_cgs:.6e} cm   (1 code length unit)")
print(f"  box_width_code        = {box_width_code:.6e} code units")
print(f"  box_width_phys_cgs    = {box_width_phys_cgs:.6e} cm")
print(f"  dx_code               = {dx_code:.6e} code units / cell")
print(f"  dx_phys_cgs           = {dx_phys_cgs:.6e} cm / cell")
print("  --- code -> cgs scales ---")
print(f"  L_cgs                 = {cu.L_cgs:.6e} cm")
print(f"  V_cgs                 = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs                 = {cu.T_cgs:.6e} s")
print(f"  light_speed_code      = {light_speed_code:.6f}")
print(f"  time_code             = {time_code:.6e}")
print(f"  source_rate_code      = {source_rate_code:.4e} photons / code time")
print("  --- time step ---")
print(f"  dt_cfl (expected)     = {dt_cfl:.6e} code = {dt_cfl * cu.T_cgs:.4e} s")
print(f"  max_dt                = {max_dt:.6e} code")
print(f"  estimated n_steps     = {n_steps_est}   (n_super_step = {n_super_step})")
print(f"  expected front  c*t   = {c_cgs * t_phys:.4e} cm "
      f"= {c_cgs * t_phys / dx_phys_cgs:.1f} cells "
      f"= {c_cgs * t_phys / box_width_phys_cgs:.3f} box widths")
print("=" * 70)

# ============================================================================
# SOLVER
# ============================================================================

# eq.eps is an ABSOLUTE floor in code units (jnp.maximum(E, eps) in
# get_conservatives_from_primitives). Since E_code = photons_per_cell /
# dx_code**3, changing unit_length changes the field amplitude and therefore
# the part of the profile that gets clipped by eps: eps must stay << typical E_code.
eps_code = float(os.environ.get("EPS", 1e-20))
eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    debug=False,
)
eq_test_hydro = dh.EquationManager(
    gamma=5.0 / 3.0,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    n_cons=6,
    passive_names=("x_HII",),
)
assert abs(cfl_code - eq_test.cfl) < 1e-12, (
    f"desynchronized cfl: dt_cfl computed with {cfl_code}, solver has {eq_test.cfl}"
)
# order of magnitude of the density injected per time step, to compare with eps
source_density_per_step = source_rate_code * dt_cfl / cell_volume_code
print(f"  eps_code              = {eps_code:.3e}   "
      f"(source/step = {source_density_per_step:.3e} [ph/vol code], "
      f"ratio = {source_density_per_step / eps_code:.2e})")
if source_density_per_step < 1e4 * eps_code or source_density_per_step < 1e-5:
    print("  !! WARNING: the field amplitude in code units is low. "
          "The solver's absolute thresholds (eps, the +1e-30 values) and float32 start to affect "
          "the profile: measured at N=64, a field around ~1e-6 in code units loses "
          "~10% on the peak (the total photons remain exact). "
          "Use a larger unit_length (smaller dx_code) or lower EPS.")
solver_test = dh.LaxFriedrichs_Radiative_transfer(
    equation_manager=eq_test, signal_speed=dh.signal_speed_Rusanov
)
solver_test_hydro = dh.LaxFriedrichs(
    equation_manager=eq_test_hydro, signal_speed=dh.signal_speed_Rusanov
)

class StateBlockFlux:
    def __init__(self, base_flux, state_slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o

    def flux(self, sol, ax, params, flux):
        local_sol = sol[self.state_slice]
        local_flux = self.base_flux.flux(local_sol, ax, params, flux)
        full_flux = jnp.zeros_like(sol)
        return full_flux.at[self.state_slice].set(local_flux)

    def timestep(self, sol):
        return self.base_flux.timestep(sol[self.state_slice])

# dx_code must be provided here: it sets the CFL of the RT flux.
rt_flux = StateBlockFlux(
    dh.ConvectiveFlux_Radiative_transfer(
        eq_test, solver_test, dh.PLM(limiter="VANLEER"), dx=dx_code,
    ),
    slice(0, eq_test.n_cons),
)
hydro_flux = StateBlockFlux(
    dh.ConvectiveFlux(
        eq_test_hydro, solver_test_hydro, dh.PLM(limiter="VANLEER"), dx=dx_code,
    ),
    slice(eq_test.n_cons, eq_test.n_cons + eq_test_hydro.n_cons),
)
# n_cons_total = (EquationManager_RT.get_conservatives_from_primitives() +
#                 dh.EquationManager.get_conservatives_from_primitives())

stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=dx_code,                    # cell volume: rate [ph/t] -> density [ph/vol]
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=True,
    gaussian_star=True,            # gaussian_star=False breaks under jit (python `if` on tracer)
    injection_geometry="3D",
    eq=eq_test,
    hydro_eq=eq_test_hydro,
    debug=False,
    momentum_only=False,
    chemistry=True,
    cu=cu,
    chemistry_case="A",         # alpha^A in the chemistry <-> b_rec = 1 in eq. 25'
    xHII_weighted=False,        # x_HII is stored as a gas scalar, weighted by rho in conservatives
    X_H=1.0,                    # pure hydrogen
    chem_max_frac=0.9,          # |Delta N| <= 90% of N per step
)
heatcool_force = dh.physics.cooling.HeatCoolForce_basic(
    eq=eq_test,
    hydro_eq=eq_test_hydro,
    cu=cu,
    light_speed=light_speed_code,
    case="A",                   # eta^A, consistent with alpha^A above
    expansion_factor=1.0,       # 'a' in the Compton term
    xHII_weighted=False,
    X_H=1.0,
    # <h nu> = 13.6 eV -> photoheating is IDENTICALLY zero (monochromatic
    # threshold group). Set e.g. 18.0 to get a physically heated HII region.
    mean_photon_energy_eV=13.6,
)
ionization_force = HydrogenIonizationForce(
    stellar_force,
    case="A",
    collisional=True,
    max_frac=0.9,
)

# coupled_eq = EquationManagerCoupled(hydro_eq=eq_test_hydro, rt_eq=eq_test)
# n_cons_total = eqmanagerrtgetcons +eqmanagergasgetcons   # 5 + 4 + 1 = 10



hydrosim_test = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar_force, heatcool_force, ionization_force],#heatcool_force
    dx=dx_code,                    # flux divergence: rhs / dx_o
    max_dt=max_dt,
    
)
assert hydrosim_test.dx_o == rt_flux.dx_o == stellar_force.dx, "desynchronized dx !"
print("hydrosim_test.dx_o =", hydrosim_test.dx_o, " rt_flux.dx_o =", rt_flux.dx_o,
      " force.dx =", stellar_force.dx, " cfl =", eq_test.cfl)
print("expected dt_code   =", eq_test.cfl / (3.0 * light_speed_code / dx_code))

params = {
    "star_masses":        jnp.array([1.0]),
    "star_ages":          jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions":     jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}
# sol_test = jnp.zeros((10, size_shape, size_shape, size_shape), dtype=jnp.float32)
# center = size_shape // 2
# sol_test = sol_test.at[0, center, center, center].set(1e-20)  # RT photon density
# sol_test = sol_test.at[5, center, center, center].set(1.0)      # hydro density
# sol_test = sol_test.at[9, center, center, center].set(1.0)      # hydro pressure
sol_test = jnp.zeros((10, size_shape, size_shape, size_shape), dtype=jnp.float32)
center = size_shape // 2

# --- Ambient neutral hydrogen medium, uniform everywhere (Stromgren setup) ---
# IMPORTANT: hydro.evolve_*() treats its input as the CONSERVATIVE state
# (that is what _hydrostep / rhs_unsplit / ConvectiveFlux assume), so slot 9
# must hold E_tot = rho e + 0.5 rho v^2, NOT the pressure, and slots 6-8 must
# hold rho*v, not v. Writing p directly in slot 9 divided the initial
# temperature by 1/(gamma-1) = 1.5.
n_H_cgs = 1.0        # cm^-3, adjust to your test case
T_ambient_K = 100.0  # K, cold neutral gas before ionization
x_HII_init = 0.0     # fully neutral initially

rho_ambient_cgs = n_H_cgs * mH_chem                # g/cm^3, pure hydrogen
rho_ambient_code = rho_ambient_cgs / cu.rho_cgs    # -> code units

# Pure hydrogen: n_tot = n_H (1 + x_HII), so p = n_tot k_B T. Using the
# fixed cu.mu = 0.61 (fully ionized primordial value) here would be
# inconsistent with the chemistry, which works with n_H and x_HII.
kB_cgs = 1.380649e-16
n_tot_cgs = n_H_cgs * (1.0 + x_HII_init)
p_ambient_cgs = n_tot_cgs * kB_cgs * T_ambient_K
p_ambient_code = p_ambient_cgs / cu.P_cgs
Etot_ambient_code = p_ambient_code / (5.0 / 3.0 - 1.0)   # v = 0 -> E_tot = e_th

sol_test = sol_test.at[5].set(rho_ambient_code)      # rho
sol_test = sol_test.at[8].set(Etot_ambient_code)     # E_tot (conservative!)
sol_test = sol_test.at[9].set(rho_ambient_code * x_HII_init)  # rho * x_HII
sol_test = sol_test.at[0, center, center, center].set(1e-20)  # RT photon seed
print(f"\nInitial gas: n_H = {n_H_cgs:.3e} cm^-3, T = {T_ambient_K:.1f} K, "
      f"x_HII = {x_HII_init:.2f}")
print(f"  rho_code = {rho_ambient_code:.4e}, p_code = {p_ambient_code:.4e}, "
      f"Etot_code = {Etot_ambient_code:.4e}")

# --- float32 dynamic-range check -------------------------------------------
# The state is float32 and every code-unit amplitude scales with the unit
# system: sol[0] is a photon density PER CODE VOLUME (so it scales as
# L_cgs^3), while rho_code scales as L_cgs^3 / M_cgs. Both blow past the
# float32 ceiling (3.4e38) or sink below its floor (1.2e-38) as soon as a
# realistic astrophysical unit length is used. Sizing the MASS unit so that
# rho_code ~ 1 fixes the hydro side; the RT side needs float64.
_f32_max, _f32_min = 3.4e38, 1.2e-38
_N_peak_code = source_rate_code * dt_cfl / cell_volume_code
_range_vals = {"rho_code": rho_ambient_code, "Etot_code": Etot_ambient_code,
               "N_gamma_code (peak/step)": _N_peak_code}
_bad = {k: v for k, v in _range_vals.items()
        if v != 0.0 and (abs(v) > 1e-3 * _f32_max or abs(v) < 1e3 * _f32_min)}
if _bad:
    print("  !! WARNING: float32 dynamic range at risk for "
          + ", ".join(f"{k} = {v:.2e}" for k, v in _bad.items()))
    m_suggest = rho_ambient_cgs * unit_length_phys_cgs ** 3
    print(f"     Pick the MASS unit so that rho_code ~ 1, i.e. "
          f"mass = {m_suggest:.3e} g (= {m_suggest / 1.98847e33:.3e} Msun),")
    print("     and enable float64 "
          "(jax.config.update('jax_enable_x64', True) + sol in float64) "
          "if the photon density in code units is large.")
print(f"  sigma_HI(nu_0) = {hchem.SIGMA_HI_0_CGS:.4e} cm^2, "
      f"mean free path = {1.0 / (n_H_cgs * hchem.SIGMA_HI_0_CGS):.4e} cm "
      f"= {1.0 / (n_H_cgs * hchem.SIGMA_HI_0_CGS) / dx_phys_cgs:.2e} cells")
alpha_B_1e4 = float(hchem.alpha_B_HII_cgs(1.0e4))
r_stromgren = (3.0 * source_rate_phys / (4.0 * np.pi * alpha_B_1e4 * n_H_cgs ** 2)) ** (1.0 / 3.0)
print(f"  alpha_B(1e4 K) = {alpha_B_1e4:.4e} cm^3/s -> "
      f"R_Stromgren = {r_stromgren:.4e} cm = {r_stromgren / dx_phys_cgs:.2e} cells")
t_rec = 1.0 / (alpha_B_1e4 * n_H_cgs)
print(f"  recombination time 1/(alpha_B n_H) = {t_rec:.4e} s "
      f"(run duration {t_phys:.3e} s)")

# ---------------------------------------------------------------------------
# CHEMISTRY CONSISTENCY CHECK
# The chemistry only does something if the box is optically thick, if the
# Stromgren radius fits inside the box, and if the run is long enough for
# the ionization/recombination timescales. All three fail badly for a 3 cm
# box at n_H = 1 cm^-3 over 5e-11 s, so x_HII stays ~1e-18 no matter how
# correct the implementation is.
# ---------------------------------------------------------------------------
tau_box = n_H_cgs * hchem.SIGMA_HI_0_CGS * box_width_phys_cgs
Gamma_typ = (hchem.SIGMA_HI_0_CGS * c_cgs * source_rate_phys
             / (4.0 * np.pi * c_cgs * max(dx_phys_cgs, 1e-30) ** 2))  # ~ at 1 cell
print("  --- chemistry consistency ---")
print(f"  optical depth across the box tau = n_H sigma L = {tau_box:.3e}")
print(f"  R_Stromgren / box                 = {r_stromgren / box_width_phys_cgs:.3e}")
print(f"  t_run / t_recombination           = {t_phys / t_rec:.3e}")
print(f"  t_run / t_ionization (~1 cell)    = {t_phys * Gamma_typ:.3e}")
if tau_box < 0.1 or t_phys / t_rec < 1e-3 or r_stromgren > box_width_phys_cgs:
    n_H_needed = 1.0 / (hchem.SIGMA_HI_0_CGS * box_width_phys_cgs)
    print("  !! WARNING: this setup CANNOT show any ionization physics.")
    print(f"     The gas is optically thin (tau = {tau_box:.1e} << 1), the Stromgren "
          f"sphere is {r_stromgren / box_width_phys_cgs:.1e} box widths wide, and the run "
          f"lasts {t_phys / t_rec:.1e} recombination times.")
    print(f"     For tau ~ 1 at this box size you would need n_H ~ {n_H_needed:.2e} cm^-3;")
    print("     a standard Stromgren test instead uses BOXPHYS ~ '2e19 cm' (~6 pc), "
          "n_H ~ 1e3 cm^-3, SRC ~ 5e48 ph/s and TPHYS ~ '1e12 s' (~30 kyr).")
    print("     The RT transport test (free streaming) is unaffected.")
print(f"\nRunning to t = {t_phys:.3e} s = {time_code:.3e} code units ...")
# Field names of the CONSERVATIVE state actually carried by the solver.
field_names = ["E_gamma", "Fx", "Fy", "Fz",
               "rho", "rho_vx", "rho_vy", "rho_vz", "E_tot", "rho_xHII"]
print("=== Pre-flight check on sol_test (before evolve) ===")
for k, name in enumerate(field_names):
    arr = np.asarray(sol_test[k])
    n_nan = np.sum(np.isnan(arr))
    print(f"  {name:15s} min={arr.min():.3e} max={arr.max():.3e} n_nan={n_nan}")

rho_check = np.asarray(sol_test[4])
p_check = np.asarray(sol_test[8])
print(f"  rho zeros: {np.sum(rho_check == 0)}/{rho_check.size}")
print(f"  p   zeros: {np.sum(p_check == 0)}/{p_check.size}")
#RUNNING TILL TIME
field_test, _, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test), params, time_code
)
# NOTE: this block used to re-print sol_test (the INPUT) instead of
# field_test, so it could never detect anything going wrong during evolve.
print("=== Post-run check on field_test (conservative state) ===")
for k, name in enumerate(field_names):
    arr = np.asarray(field_test[k])
    n_nan = np.sum(np.isnan(arr))
    print(f"  {name:15s} min={arr.min():.3e} max={arr.max():.3e} n_nan={n_nan}")

# Physical (primitive) diagnostics derived from the conservative state.
hydro_block = np.asarray(field_test[eq_test.n_cons:], dtype=np.float64)
hydro_prim = eq_test_hydro.get_primitives_from_conservatives(hydro_block)
_rho = np.asarray(hydro_prim[0], dtype=np.float64)
_mom2 = sum(np.asarray(hydro_block[k], dtype=np.float64) ** 2 for k in (1, 2, 3))
# NB: no absolute 1e-30 floor here -- with L = 1 cm and n_H = 1 cm^-3 the
# thermal energy is ~2e-35 in code units, i.e. FAR below 1e-30, and such a
# floor would report T ~ 4e6 K instead of 100 K.
_tiny = np.finfo(np.float64).tiny
_p_code = (5.0 / 3.0 - 1.0) * np.maximum(
    np.asarray(field_test[8], dtype=np.float64) - 0.5 * _mom2 / np.maximum(_rho, _tiny),
    _tiny,
)
_x = np.clip(np.asarray(hydro_prim[5], dtype=np.float64), 0.0, 1.0)
_n_H = _rho * cu.rho_cgs / mH_chem
_T_K = _p_code * cu.P_cgs / (np.maximum(_n_H * (1.0 + _x), _tiny) * kB_cgs)
print(f"  p_code   min/max = {_p_code.min():.3e} / {_p_code.max():.3e}")
print(f"  T [K]    min/max = {_T_K.min():.3e} / {_T_K.max():.3e}  (initial {T_ambient_K:.1f} K)")
print(f"  n_H[cm^-3] min/max = {_n_H.min():.3e} / {_n_H.max():.3e}")
print(f"  x_HII    min/max/mean = {_x.min():.3e} / {_x.max():.3e} / {_x.mean():.3e}")

dt_hist = np.asarray(dt_hist)
dt_sum  = float(dt_hist[dt_hist > 0].sum())
n_steps = int(n_steps)
print("Done.")
print(f"  steps           = {n_steps},  dt_code = {dt_hist[0]:.6e}"
      f"  (dt_cfl = {dt_cfl:.6e})")
print(f"  sum(dt)         = {dt_sum:.6e} code = {dt_sum * cu.T_cgs:.4e} s"
      f"  (target {t_phys:.4e} s)")
if n_steps >= n_super_step:
    print(f"  !! WARNING: n_steps saturated n_super_step={n_super_step}: "
          f"t_target is NOT reached. Increase NSTEP.")
if dt_hist[0] < 0.99 * dt_cfl:
    print(f"  !! WARNING: dt ({dt_hist[0]:.3e}) < dt_cfl ({dt_cfl:.3e}): "
          f"max_dt={max_dt:.3e} caps the CFL.")

# ============================================================================
# DIAGNOSTICS
# ============================================================================
# E3d      : code field, photon DENSITY [photons / code volume]
# E_cell   : photons per cell (convention invariant) = E3d * dx_code**3
# E_dens   : physical density [photons / cm^3]        = E_cell / dx_phys_cgs**3

# float64 for diagnostics: the solver field is in float32, and multiplying
# by dx_code**3 (1.95e-6 at N=256) would underflow the tail of the profile.
E3d    = np.asarray(field_test[0], dtype=np.float64)
E_cell = E3d * cell_volume_code
c      = size_shape // 2

print(f"  E_code min/max  = {E3d.min():.4e} / {E3d.max():.4e}  [ph / vol code]")
print(f"  E_cell min/max  = {E_cell.min():.4e} / {E_cell.max():.4e}  [ph / cell]")
photons_in_box = E_cell.sum()
photons_expect = source_rate_code * dt_sum
print(f"  photons in box  = {photons_in_box:.6e}   expected = {photons_expect:.6e}"
      f"   ratio = {photons_in_box / max(photons_expect, 1e-300):.6f}")
print(f"  photons in box  = {photons_in_box:.6e} ph "
      f"= {source_rate_phys * dt_sum * cu.T_cgs:.6e} ph expected (cgs)")

line = E_cell[c:, c, c]
peak = E_cell.max()
for th in [1e-3, 1e-6, 1e-10, 1e-15]:
    idx = np.where(line > peak * th)[0]
    r = idx.max() if idx.size else 0
    print(f"  thr {th:.0e} of peak -> radius {r:3d} cells = {r * dx_phys_cgs:.4e} cm "
          f"= {r * dx_code:.4e} code units")
print(f"  expected free-streaming radius = {c_cgs * t_phys / dx_phys_cgs:.1f} cells")


if field_test.shape[0] > eq_test.n_active:
    xHII_3d = np.asarray(hydro_prim[eq_test_hydro.xHII_id], dtype=np.float64)

    print(f"  x_HII min/max/mean = "
          f"{xHII_3d.min():.4e} / {xHII_3d.max():.4e} / {xHII_3d.mean():.4e}")

    # vérification des bornes physiques [0,1] -- doit toujours être vrai
    n_out_of_bounds = np.sum((xHII_3d < -1e-9) | (xHII_3d > 1.0 + 1e-9))
    print(f"  x_HII cells out of [0,1] bounds = {n_out_of_bounds}")

    # cohérence avec le front radiatif: x_HII doit être ~1 près de la
    # source et retomber vers 0 loin devant le front de photons
    line_xHII = xHII_3d[c:, c, c]
    print(f"  x_HII along +x axis (first 10 cells) = {line_xHII[:10]}")
    print(f"  x_HII along +x axis (last 10 cells)  = {line_xHII[-10:]}")
else:
    print("  !! x_HII not present in field_test (n_cons/passive_names not configured)")

# ============================================================================
# PLOT
# ============================================================================
def compute_extent_phys(size_shape, dx_phys=dx_phys_cgs, centered=True):
    """
    Converts pixel indices [0, size_shape] into physical units (cm):
    one pixel = dx_phys cm. centered=True centers the origin on the source
    (cell size_shape//2), centered=False keeps pixel 0 -> 0.
    """
    box_extent = size_shape * dx_phys
    if centered:
        half = box_extent / 2.0
        return [-half, half, -half, half]
    return [0, box_extent, 0, box_extent]

plt.style.use("dark_background")
E_slice = E_cell[:, :, c]          # photons per cell
extent  = compute_extent_phys(size_shape, centered=False)
fig, ax = plt.subplots(figsize=(6, 5))
pos = E_slice[E_slice > 0]
if pos.size == 0:
    print("  !! WARNING: E_slice has no positive values -- skipping this plot "
          "(E_gamma is zero everywhere on this slice, injection likely failed).")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot",
                   extent=extent, norm=LogNorm(vmin=max(pos.min(), peak * 1e-12), vmax=peak))
    ax.set_xlabel("y [cm]"); ax.set_ylabel("x [cm]")
    ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
    fig.colorbar(im, ax=ax, label="photons per cell")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("wrote", out)
    

# raw solver field = density in code units, axes in cell indices
fig, ax = plt.subplots(figsize=(6, 5))
E_slice_code = E3d[:, :, c]
im = ax.imshow(np.ma.masked_less_equal(E_slice_code, 0.0), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"E_gamma code, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("wrote", out)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log10(np.ma.masked_less_equal(E_slice_code, 0.0)), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"log10 E_gamma code, t = {t_phys:.2e} s  "
             f"(ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
fig.colorbar(im, ax=ax, label="log10 photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut_log.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("wrote", out)


xHII_slice = xHII_3d[:, :, c]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(xHII_slice, origin="lower", cmap="viridis",
               extent=extent, vmin=0.0, vmax=1.0)
ax.set_xlabel("y [cm]"); ax.set_ylabel("x [cm]")
ax.set_title(f"Ionization fraction x_HII, t = {t_phys:.2e} s")
fig.colorbar(im, ax=ax, label=r"$x_{HII}$")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"xHII_slice_{run_tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("wrote", out)

# ============================================================================
# NEW FIGURE: photon density in cm^-3 in the colorbar
# ============================================================================
# Conversion "photons per cell" -> "photons per cm^3":
#   n_gamma [cm^-3] = N_gamma [photons/cell] / dx_phys_cgs^3 [cm^3]
#                   = E_code / L_cgs^3          (the two routes agree)
E_slice_density = E_slice / cell_volume_cm3   # photons / cm^3
# the two conversion routes must agree (up to machine precision), since both
# cell_volume_cm3 and cu.L_cgs derive from the SAME unit_length_phys_cgs.
n_nan_density = np.sum(np.isnan(E_slice_density))
n_nan_code = np.sum(np.isnan(E_slice_code))
print(f"  NaN count: E_slice_density={n_nan_density}, E_slice_code={n_nan_code}")
print(f"  cu.L_cgs               = {cu.L_cgs:.15e} cm")
print(f"  unit_length_phys_cgs   = {unit_length_phys_cgs:.15e} cm")
print(f"  dx_phys_cgs            = {dx_phys_cgs:.15e} cm")
print(f"  dx_code * cu.L_cgs     = {dx_code * cu.L_cgs:.15e} cm")

diff = np.abs(E_slice_density - E_slice_code / cu.L_cgs**3)
print(f"  max abs diff = {np.nanmax(diff):.3e}")

assert np.allclose(E_slice_density, E_slice_code / cu.L_cgs**3,
                    rtol=1e-9, atol=1e-15 * np.nanmax(E_slice_density), equal_nan=True)
assert np.allclose(E_slice_density, E_slice_code / cu.L_cgs**3,
                   rtol=1e-9, atol=1e-15 * E_slice_density.max())

pos_density = E_slice_density[E_slice_density > 0]
if pos_density.size == 0:
    print("  !! WARNING: E_slice_density has no positive values -- skipping this plot.")
else:
    peak_density = pos_density.max()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.ma.masked_less_equal(E_slice_density, 0.0),
        origin="lower", cmap="hot",
        extent=extent,
        norm=LogNorm(vmin=max(pos_density.min(), peak_density * 1e-12), vmax=peak_density),
    )
    ax.set_xlabel("y [cm]")
    ax.set_ylabel("x [cm]")
    ax.set_title(f"Photon number density, t = {t_phys:.2e} s")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$n_\gamma$  [photons cm$^{-3}$]")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_density_cm3_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("wrote", out)


assert field_test.shape[0] == len(field_names), (
    f"field_test has {field_test.shape[0]} fields, expected {len(field_names)}"
)

BASE_FIELDS_DIR = os.path.join(BASE_OUTPUT_DIR, "fields")
os.makedirs(BASE_FIELDS_DIR, exist_ok=True)

size_x = field_test.shape[1]
c = size_x // 2  # slice index through the box center

for k, name in enumerate(field_names):
    field_dir = os.path.join(BASE_FIELDS_DIR, name)
    os.makedirs(field_dir, exist_ok=True)

    data_3d = np.asarray(field_test[k], dtype=np.float32)
    data_slice = data_3d[:, :, c]

    # ── imshow of the slice ──
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(np.log10(data_slice), origin="lower", cmap="hot")
    ax.set_xlabel("y cell")
    ax.set_ylabel("x cell")
    ax.set_title(f"{name} slice (z={c}) - {run_tag}")
    fig.colorbar(im, ax=ax, label=name)
    plt.tight_layout()
    plt.savefig(f"{field_dir}/slice_{name}_{run_tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── full 3D cube saved as CSV (flattened with cell indices) ──
    xi, yi, zi = np.meshgrid(
        np.arange(data_3d.shape[0]),
        np.arange(data_3d.shape[1]),
        np.arange(data_3d.shape[2]),
        indexing="ij",
    )
    csv_path = os.path.join(field_dir, f"cube_{name}_{run_tag}.csv")
    header = "x,y,z," + name
    np.savetxt(
        csv_path,
        np.column_stack([xi.ravel(), yi.ravel(), zi.ravel(), data_3d.ravel()]),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", "%d", "%.8e"],
    )

    print(f"[{name}] wrote slice PNG and CSV in {field_dir}")

# ============================================================================
# SPHERICAL AVERAGE + POWER-LAW (log-log linear) REGRESSION
# ============================================================================
# r is in CELLS (indices). r_phys = r * dx_phys_cgs [cm], r_code = r * dx_code.
# The fitted exponent b is invariant under a change of units (constant factor).

# def spherical_average(field_3d, center):
#     cx, cy, cz = center
#     nx, ny, nz = field_3d.shape
#     x = np.arange(nx) - cx
#     y = np.arange(ny) - cy
#     z = np.arange(nz) - cz
#     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#     R = np.sqrt(X**2 + Y**2 + Z**2)
#     r_int = np.round(R).astype(int)
#     r_max = int(r_int.max())

#     r_vals, avg_vals = [], []
#     for r in range(r_max + 1):
#         mask = r_int == r
#         vals = field_3d[mask]
#         finite = vals[np.isfinite(vals) & (vals > 0.0)]
#         if finite.size > 0:
#             r_vals.append(r)
#             avg_vals.append(float(np.mean(finite)))

#     return np.array(r_vals, dtype=float), np.array(avg_vals, dtype=float)


# def analyze_inverse_r2(field_3d, size_shape, cell_size_phys, tag,
#                         radius_truncation=None,
#                         output_dir=BASE_OUTPUT_DIR):
#     center_idx = size_shape // 2
#     sigma = max(1, round(size_shape // 100))
#     injection_radius = len(jnp.arange(-3 * sigma, 3 * sigma + 1)) // 2

#     if radius_truncation is None:
#         radius_truncation = max(injection_radius + 8, size_shape)

#     r_sph, y_sph = spherical_average(
#         np.array(field_3d, dtype=float),
#         center=(center_idx, center_idx, center_idx)
#     )

#     mask = (r_sph > injection_radius) & (r_sph < radius_truncation)
#     r_valid = r_sph[mask]
#     y_valid = y_sph[mask]
#     x_valid = center_idx + r_valid

#     if r_valid.size < 5:
#         print(f"[{tag}] Not enough valid points for 1/r^2 analysis.")
#         return None

#     log_r = np.log(r_valid)
#     log_y = np.log(y_valid)

#     def line_model(x, c, b):
#         return c - b * x

#     popt, pcov = curve_fit(
#         line_model, log_r, log_y, p0=[log_y[0], 2.0], maxfev=20000
#     )

#     c_fit, b = float(popt[0]), float(popt[1])
#     b_err = float(np.sqrt(pcov[1, 1])) if pcov.size else np.nan

#     y_pred = np.exp(line_model(log_r, c_fit, b))

#     print(f"[{tag}] c={c_fit:.6f}  b={b:.6f} (+/- {b_err:.2e})")
#     print(f"[{tag}] injection_radius={injection_radius}, "
#           f"fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells "
#           f"= [{r_valid.min()*cell_size_phys:.3e}, {r_valid.max()*cell_size_phys:.3e}] cm")

#     os.makedirs(output_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
#     ax.set_facecolor("black")
#     ax.plot(log_r, log_y, "o", color="white", ms=3, label="Spherical avg")
#     ax.plot(log_r, np.log(y_pred), "r-", lw=2, label=f"fit a*r^(-b), b={b:.3f}")
#     ax.axvline(np.log(injection_radius), color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
#     ax.set_xlabel("log(r [cell])")
#     ax.set_ylabel("log(Shell-averaged field)")
#     ax.set_title(f"log-log spherical average - {tag}")
#     ax.legend(fontsize=8)
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/loglog_spherical_average_{tag}.png", dpi=300, bbox_inches="tight")
#     plt.show()

#     fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
#     ax.set_facecolor("black")
#     ax.plot(x_valid, y_valid, "o", color="0.75", ms=3, label="Spherical avg data")
#     ax.plot(x_valid, y_pred, "r-", lw=2, label=f"fit a*r^(-b), b={b:.3f}")
#     ax.axvline(center_idx + injection_radius, color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
#     ax.set_xlabel("Cell index (center + r)")
#     ax.set_ylabel("Shell-averaged field value")
#     ax.set_title(f"Linear spherical average - {tag}")
#     ax.legend(fontsize=8)
#     plt.show()
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/linear_spherical_average_{tag}.png", dpi=300, bbox_inches="tight")

#     return {
#         "tag": tag, "r_sph": r_sph, "y_sph": y_sph, "r_valid": r_valid,
#         "y_valid": y_valid, "x_valid": x_valid, "y_pred": y_pred,
#         "c": c_fit, "b": b, "b_err": b_err,
#         "injection_radius": injection_radius, "cell_size_phys": cell_size_phys,
#     }


# def value_average_radius(field_3d, size_shape, cell_size_phys, tag,
#                           radius_truncation=None,
#                           output_dir=BASE_OUTPUT_DIR):
#     center_idx = size_shape // 2
#     sigma = max(1, round(size_shape // 100))
#     injection_radius = len(jnp.arange(-3 * sigma, 3 * sigma + 1)) // 2

#     if radius_truncation is None:
#         radius_truncation = max(injection_radius + 8, size_shape)

#     r_sph, y_sph = spherical_average(
#         np.array(field_3d, dtype=float),
#         center=(center_idx, center_idx, center_idx)
#     )

#     mask = (r_sph > injection_radius) & (r_sph < radius_truncation)
#     r_valid = r_sph[mask]
#     y_valid = y_sph[mask]
#     x_valid = center_idx + r_valid

#     if r_valid.size < 5:
#         print(f"[{tag}] Not enough valid points for 1/r^2 analysis.")
#         return None

#     log_y = np.log(y_valid)

#     print(f"[{tag}] injection_radius={injection_radius}, "
#           f"fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells")

#     os.makedirs(output_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
#     ax.set_facecolor("black")
#     ax.semilogx(r_valid, log_y, "o", color="white", ms=3, label="Spherical avg")
#     ax.axvline(np.log(injection_radius), color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
#     ax.set_xlabel("log(r [cell])")
#     ax.set_ylabel("log(Shell-averaged field)")
#     ax.set_title(f"log-log spherical average - {tag}")
#     ax.legend(fontsize=8)
#     ax.grid(which="both", color="0.25", ls="--", lw=0.5)
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/loglog_spherical_average_{tag}_brut.png", dpi=300, bbox_inches="tight")
#     plt.show()

#     fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
#     ax.set_facecolor("black")
#     ax.plot(x_valid, y_valid, "o", color="0.75", ms=3, label="Spherical avg data")
#     ax.axvline(center_idx + injection_radius, color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
#     ax.set_xlabel("Cell index (center + r)")
#     ax.set_ylabel("Shell-averaged field value")
#     ax.set_title(f"Linear spherical average - {tag}")
#     ax.legend(fontsize=8)
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/linear_spherical_average_{tag}_brut.png", dpi=300, bbox_inches="tight")
#     plt.show()
#     return {
#         "tag": tag, "r_sph": r_sph, "y_sph": y_sph, "r_valid": r_valid,
#         "y_valid": y_valid, "x_valid": x_valid,
#         "injection_radius": injection_radius, "cell_size_phys": cell_size_phys,
#     }


# def clamp_truncation(r_trunc, label):
#     """Beyond N/2 the spherical shells leave the periodic box."""
#     r_max = size_shape // 2
#     if r_trunc > r_max:
#         print(f"  [{label}] radius_truncation {r_trunc} > N/2 = {r_max}: "
#               f"clamped to {r_max} (shells outside the box).")
#         return r_max
#     return r_trunc


# evolve_value_radius_result = value_average_radius(
#     field_3d=E_cell,
#     size_shape=size_shape,
#     cell_size_phys=dx_phys_cgs,
#     tag=run_tag,
#     radius_truncation=clamp_truncation(int(os.environ.get("RTRUNC_AVG", 250)), "avg"),
#     output_dir=BASE_OUTPUT_DIR,
# )
# fit_result = analyze_inverse_r2(
#     field_3d=E_cell,
#     size_shape=size_shape,
#     cell_size_phys=dx_phys_cgs,
#     tag=run_tag,
#     radius_truncation=clamp_truncation(int(os.environ.get("RTRUNC_FIT", 90)), "fit"),
#     output_dir=BASE_OUTPUT_DIR,
# )

# if fit_result is not None:
#     print(f"\nPower-law fit result: n_gamma(r) ~ r^(-{fit_result['b']:.3f})  "
#           f"(+/- {fit_result['b_err']:.2e})")
# else:
#     print("\nPower-law fit could not be performed (not enough valid points).")
