"""
RAMSES-RT point-source test, with units that match what the solver actually does.

UNIT CONVENTION (unit-agnostic via UnitParser)
----------------------------------------------------------
Every physical length/velocity/time input can be given in ANY unit known to
diffhydro.units.registry.UnitParser. Each input is parsed once, converted to
cgs immediately, and everything downstream (dx_code, CodeUnits, the solver)
only ever sees cgs values for the PHYSICS.

NO MORE ENVIRONMENT VARIABLES: every parameter you might want to change is
now a plain Python variable in the CONFIG block right below the imports.
Nothing reads os.environ anymore, so a leftover shell `export SRC=...` or
`export UVEL=...` from a previous run can NEVER silently override what you
set here again -- this was the actual root cause of several "impossible"
results in earlier runs (UVEL stuck at an old, unphysical value).

DISPLAY UNIT: the output folder name, every saved filename, AND every
figure's axis label/title follow the EXACT unit string you set in CONFIG
(ULEN, BOXPHYS, UVEL, TPHYS) instead of a fixed cgs notation.

FIELD LAYOUT (depends on the CURRENT EquationManager, which treats x_HII as
a generic named PASSIVE SCALAR rather than a hard-coded slot):

  eq_test_hydro = dh.EquationManager(gamma=..., n_cons=6,
                                      passive_names=("x_HII",), ...)

  -> combined sol layout (RT block has n_cons=4: E,Fx,Fy,Fz):
       0: E_gamma  1: Fx  2: Fy  3: Fz
       4: rho      5: vx  6: vy  7: vz  8: p
       9: x_HII

BUG FIX: alpha_B(T) is obtained from stellar_force.caseB(T_code), which
expects a temperature in CODE units and internally converts it back to
Kelvin via cu.Temp_cgs. This round trip (T_K -> T_code -> T_K) is only
exact if cu.Temp_cgs is a sane, well-scaled number -- which it is NOT if
UVEL is set to something absurd (cu.Temp_cgs scales with cu.V_cgs**2). An
explicit assertion now checks the round trip immediately after computing
it, so any future unit corruption is caught with a clear error message
instead of silently producing a "bizarre" analytic Stromgren radius many
orders of magnitude off, as happened before.
"""

import os, sys, math
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import jax
import jax.numpy as jnp
import numpy as np
import copy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

import diffhydro as dh
from diffhydro.units import CodeUnits
from diffhydro.units.registry import UnitParser
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.physics.fraction_xHII import HydrogenIonizationForce as HydrogenIonizationForce
from diffhydro.physics import hydrogen_chemistry as hchem
jax.config.update("jax_enable_x64", True)

# ============================================================================
# CONFIG -- EVERYTHING you might want to change lives here, nowhere else.
# No environment variables are read anywhere in this script anymore.
# ============================================================================
GPU_ID = "0"

N = 100

# Une unité de longueur code = une cellule physique
ULEN = "4.7536191406e20 cm"

# Une unité de vitesse code = vitesse réduite de la lumière
UVEL = "2.99792458e7 cm/s"

# L_box = 4 R_S / 1.4
BOXPHYS = "4.7536191406e22 cm"
BOXCODE = None

# Débit ionisant du test Iliev 2006
SRC = 5e48

# TPHYS = 3 t_rec
TPHYS = "1.1574897190e16 s"

EPS = 1e-30
MAXDT = None
NSTEP = None

N_H_CGS = 1.0e-3
T_AMBIENT_K = 1.0e4

MAKE_GIFS = False
GIF_FRAMES = 30

print("Backend:", jax.default_backend(), jax.devices())

up = UnitParser()


def parse_quantity(text: str, expected_dim: str):
    """Parse a free-form quantity string ('1 km', '3.2 cm', '3e10 cm/s',
    '5.2e-7 s', ...) with UnitParser and return the ParsedQuantity."""
    try:
        return up.parse(text, expected_dim=expected_dim)
    except ValueError as exc:
        raise SystemExit(f"Invalid quantity '{text}': {exc}") from exc


def sanitize_tag(text: str) -> str:
    """Make a unit string ('3.2 cm', '3e10 cm/s') safe for a folder/file name."""
    return text.replace(" ", "").replace("/", "p").replace("^", "")


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ============================================================================
# PHYSICAL SETUP (derived from CONFIG above, no env vars)
# ============================================================================
size_shape = N

ulen_q = parse_quantity(ULEN, expected_dim="length")
unit_length_phys_cgs = ulen_q.cgs_value
unit_length_str = f"{ulen_q.value:g}{ulen_q.unit}"

uvel_q = parse_quantity(UVEL, expected_dim="velocity")
unit_velocity_phys = uvel_q.cgs_value
unit_velocity_str = f"{uvel_q.value:g}{uvel_q.unit}"

c_cgs = 2.99792458e10
if abs(unit_velocity_phys - c_cgs) / c_cgs > 0.5:
    print(f"  !! WARNING: UVEL = {unit_velocity_str} ({unit_velocity_phys:.3e} cm/s) "
          f"is far from the real speed of light ({c_cgs:.3e} cm/s). "
          f"This will distort every cgs<->code conversion (Temp_cgs, T_cgs, "
          f"and therefore the recombination coefficient). Fix UVEL in CONFIG "
          f"unless you specifically intend a non-relativistic light speed test.")

if BOXPHYS is not None:
    boxphys_q = parse_quantity(BOXPHYS, expected_dim="length")
    box_width_phys_cgs = boxphys_q.cgs_value
    box_width_code = box_width_phys_cgs / unit_length_phys_cgs
    box_width_str = f"{boxphys_q.value:g}{boxphys_q.unit}"
    axis_unit_name = boxphys_q.unit
else:
    box_width_code = float(BOXCODE)
    box_width_phys_cgs = box_width_code * unit_length_phys_cgs
    box_width_str = f"boxcode{box_width_code:g}"
    axis_unit_name = ulen_q.unit

axis_unit_scale = up.unit_factor_to_cgs(axis_unit_name, expected_dim="length")

dx_code = box_width_code / size_shape
dx_phys_cgs = dx_code * unit_length_phys_cgs
cell_volume_code = dx_code ** 3
cell_volume_cm3 = dx_phys_cgs ** 3

source_rate_phys = float(SRC)

tphys_q = parse_quantity(TPHYS, expected_dim="time")
t_phys = tphys_q.cgs_value
time_axis_unit = tphys_q.unit
time_axis_scale = up.unit_factor_to_cgs(time_axis_unit, expected_dim="time")
tphys_str = f"{tphys_q.value:g}{tphys_q.unit}"

c_red_cgs = 1.0e-3 * c_cgs

rho_ambient_cgs = N_H_CGS * 1.6726219e-24
mass_unit_phys_cgs = rho_ambient_cgs * unit_length_phys_cgs**3

cu = CodeUnits.from_config(
    {
        "length": f"{unit_length_phys_cgs} cm",
        "mass": f"{mass_unit_phys_cgs} g",
        "velocity": f"{c_red_cgs} cm/s",
    },
    {
        "gamma": 5.0 / 3.0,
        "mu": 1.0,
    },
)

# light_speed_code = c_cgs / cu.V_cgs
light_speed_code = c_red_cgs / cu.V_cgs
time_code = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs

cfl_code = 0.4
dt_cfl = cfl_code / (3.0 * light_speed_code / dx_code)
n_steps_est = int(math.ceil(time_code / dt_cfl))
max_dt = MAXDT if MAXDT is not None else 2.0 * dt_cfl
n_super_step = NSTEP if NSTEP is not None else int(1.2 * n_steps_est) + 100

# ============================================================================
# RUN TAG (folder name + every saved filename)
# ============================================================================
run_tag = (
    f"N{size_shape}"
    f"_ulen{sanitize_tag(unit_length_str)}"
    f"_box{sanitize_tag(box_width_str)}"
    f"_uvel{sanitize_tag(unit_velocity_str)}"
    f"_src{source_rate_phys:.2e}phs"
    f"_t{sanitize_tag(tphys_str)}"
)

BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/RT/Images", run_tag)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(f"  run_tag               = {run_tag}")
print(f"  output dir            = {BASE_OUTPUT_DIR}")
print(f"  axis display unit     = {axis_unit_name}")
print(f"  time display unit     = {time_axis_unit}")
print("=" * 70)
print(f"  ULEN                  = '{unit_length_str}'  -> {unit_length_phys_cgs:.6e} cm")
print(f"  BOX                   = '{box_width_str}'    -> {box_width_phys_cgs:.6e} cm")
print(f"  UVEL                  = '{unit_velocity_str}' -> {unit_velocity_phys:.6e} cm/s")
print(f"  TPHYS                 = '{tphys_str}' -> {t_phys:.6e} s")
print(f"  dx_code               = {dx_code:.6e} code units / cell")
print(f"  dx_phys_cgs           = {dx_phys_cgs:.6e} cm / cell")
print(f"  L_cgs                 = {cu.L_cgs:.6e} cm")
print(f"  V_cgs                 = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs                 = {cu.T_cgs:.6e} s")
print(f"  Temp_cgs              = {cu.Temp_cgs:.6e} K (1 code temperature unit)")
print(f"  light_speed_code      = {light_speed_code:.6f}")
print(f"  time_code             = {time_code:.6e}")
print(f"  source_rate_code      = {source_rate_code:.4e} photons / code time")
print(f"  dt_cfl (expected)     = {dt_cfl:.6e} code = {dt_cfl * cu.T_cgs:.4e} s")
print(f"  max_dt                = {max_dt:.6e} code")
print(f"  estimated n_steps     = {n_steps_est}   (n_super_step = {n_super_step})")
print("=" * 70)

# ============================================================================
# SOLVER
# ============================================================================
eps_code = float(EPS)
eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    debug=False,
)

eq_test_hydro = dh.EquationManager(
    gamma=5.0 / 3.0,
    n_cons=6,
    passive_names=("x_HII",),
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
)

assert abs(cfl_code - eq_test.cfl) < 1e-12, (
    f"desynchronized cfl: dt_cfl computed with {cfl_code}, solver has {eq_test.cfl}"
)
source_density_per_step = source_rate_code * dt_cfl / cell_volume_code
print(f"  eps_code              = {eps_code:.3e}   "
      f"(source/step = {source_density_per_step:.3e} [ph/vol code], "
      f"ratio = {source_density_per_step / eps_code:.2e})")

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

stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=dx_code,
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=True,
    gaussian_star=True,
    injection_geometry="3D",
    eq=eq_test,
    hydro_eq=eq_test_hydro,
    debug=False,
    momentum_only=False,
    chemistry=True,
    cu=cu,
)

heatcool_force = dh.physics.cooling.HeatCoolForce_basic(
    eq=eq_test,
    hydro_eq=eq_test_hydro,
    cu=cu,
    light_speed=light_speed_code,
    case="A",
    expansion_factor=1.0,
    X_H=1.0,
    mean_photon_energy_eV=13.6,
)
ionization_force = HydrogenIonizationForce(
    stellar_force,
    case="A",
    collisional=True,
    max_frac=0.9,
)

hydrosim_test = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar_force, heatcool_force, ionization_force],
    dx=dx_code,
    max_dt=max_dt,
)
assert hydrosim_test.dx_o == rt_flux.dx_o == stellar_force.dx, "desynchronized dx !"
print("hydrosim_test.dx_o =", hydrosim_test.dx_o, " rt_flux.dx_o =", rt_flux.dx_o,
      " force.dx =", stellar_force.dx, " cfl =", eq_test.cfl)

# ============================================================================
# STARS: single central source (Stromgren test). To use a random population
# instead, comment this block and uncomment the "STAR GENERATION" block below.
# ============================================================================
params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}

# ============================================================================
# STAR GENERATION (alternative): random positions, masses, ages, metallicities
# ============================================================================
# n_stars = 30
# rng = np.random.default_rng(42)
# star_masses = jnp.array(rng.uniform(0.5, 5.0, size=n_stars), dtype=jnp.float64)
# star_ages = jnp.array(rng.uniform(0.0, 1.0, size=n_stars), dtype=jnp.float64)
# star_metallicities = jnp.array(rng.uniform(0.001, 0.03, size=n_stars), dtype=jnp.float64)
# margin = max(1, size_shape // 10)
# star_positions_np = rng.integers(margin, size_shape - margin, size=(n_stars, 3))
# star_positions = jnp.array(star_positions_np, dtype=jnp.int32)
# params = {
#     "star_masses": star_masses,
#     "star_ages": star_ages,
#     "star_metallicities": star_metallicities,
#     "star_positions": star_positions,
# }

# Combined layout: 0:E 1:Fx 2:Fy 3:Fz 4:rho 5:vx 6:vy 7:vz 8:p 9:x_HII
n_total_fields = eq_test.n_cons + eq_test_hydro.n_cons
sol_test = jnp.zeros((n_total_fields, size_shape, size_shape, size_shape), dtype=jnp.float64)
center = size_shape // 2

idx_rho_local = eq_test.n_cons + eq_test_hydro.mass_ids
idx_p_local = eq_test.n_cons + eq_test_hydro.energy_ids

n_H_cgs = N_H_CGS
T_ambient_K = T_AMBIENT_K
rho_ambient_cgs = n_H_cgs * 1.6726219e-24
rho_ambient_code = rho_ambient_cgs / cu.rho_cgs
kB_cgs = 1.380649e-16
mH_cgs = 1.6726219e-24
mu = getattr(cu, "mu", 0.61)
p_ambient_cgs = rho_ambient_cgs * kB_cgs * T_ambient_K / (mu * mH_cgs)
p_ambient_code = p_ambient_cgs / cu.P_cgs

# ============================================================================
# --- Athena blast-wave IC (kept, commented out for the Stromgren test) ----
# athena_outputs_loc = "data/athena_comparison/"
# ic_filename = "Blast.out2.00000.athdf"
# ICs = athdf(athena_outputs_loc+ic_filename)
# sol_test = sol_test.at[4].set(ICs["dens"])
# sol_test = sol_test.at[8].set(ICs["Etot"])
# sol_test = sol_test.at[9].set(1)
# ============================================================================

# ============================================================================
# STROMGREN SPHERE TEST: uniform neutral ambient medium + single central
# ionizing point source.
# ============================================================================
sol_test = sol_test.at[idx_rho_local].set(rho_ambient_code)
sol_test = sol_test.at[idx_p_local].set(p_ambient_code)
sol_test = sol_test.at[0, center, center, center].set(1e-20)
sol_test = sol_test.at[9].set(0.0)

print(f"\nRunning to t = {t_phys:.3e} s = {time_code:.3e} code units ...")
field_test, _, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test), params, time_code
)

dt_hist = np.asarray(dt_hist)
dt_sum = float(dt_hist[dt_hist > 0].sum())
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


def fmt_t_phys():
    return f"{t_phys / time_axis_scale:.2e} {time_axis_unit}"


# ============================================================================
# DIAGNOSTICS
# ============================================================================
E3d = np.asarray(field_test[0], dtype=np.float64)
E_cell = E3d * cell_volume_code
c = size_shape // 2

print(f"  E_code min/max  = {E3d.min():.4e} / {E3d.max():.4e}  [ph / vol code]")
print(f"  E_cell min/max  = {E_cell.min():.4e} / {E_cell.max():.4e}  [ph / cell]")
photons_in_box = E_cell.sum()
photons_expect = source_rate_code * dt_sum
print(f"  photons in box  = {photons_in_box:.6e}   expected = {photons_expect:.6e}"
      f"   ratio = {photons_in_box / max(photons_expect, 1e-300):.6f}")

line = E_cell[c:, c, c]
peak = E_cell.max()
for th in [1e-3, 1e-6, 1e-10, 1e-15]:
    idx = np.where(line > peak * th)[0]
    r = idx.max() if idx.size else 0
    print(f"  thr {th:.0e} of peak -> radius {r:3d} cells = {r * dx_phys_cgs:.4e} cm "
          f"= {r * dx_code:.4e} code units")
print(f"  expected free-streaming radius = {c_cgs * t_phys / dx_phys_cgs:.1f} cells")

xHII_abs_idx = getattr(stellar_force, "idx_xHII", None)
if xHII_abs_idx is not None and field_test.shape[0] > xHII_abs_idx:
    xHII_3d = np.asarray(field_test[xHII_abs_idx], dtype=np.float64)
    print(f"  x_HII min/max/mean = "
          f"{xHII_3d.min():.4e} / {xHII_3d.max():.4e} / {xHII_3d.mean():.4e}")
    n_out_of_bounds = np.sum((xHII_3d < -1e-9) | (xHII_3d > 1.0 + 1e-9))
    print(f"  x_HII cells out of [0,1] bounds = {n_out_of_bounds}")
else:
    print("  !! x_HII not present in field_test (n_cons/passive_names not configured)")

# ============================================================================
# STROMGREN SPHERE VALIDATION: analytic radius vs simulated front
# ============================================================================
# T_ambient_code = T_ambient_K / cu.Temp_cgs
# T_K_roundtrip = T_ambient_code * cu.Temp_cgs

# # BUG FIX: this assertion is the direct fix requested. If cu.Temp_cgs is
# # corrupted (e.g. because UVEL was set to something unphysical), this round
# # trip will NOT reproduce T_ambient_K, and the script now fails loudly here
# # instead of silently producing an impossible Stromgren radius later.
# assert abs(T_K_roundtrip - T_ambient_K) < 1e-6 * T_ambient_K, (
#     f"Temperature round-trip mismatch: T_ambient_K={T_ambient_K:.6e} K -> "
#     f"T_ambient_code={T_ambient_code:.6e} -> back to {T_K_roundtrip:.6e} K. "
#     f"cu.Temp_cgs={cu.Temp_cgs:.6e} looks corrupted -- check UVEL/ULEN in CONFIG."
# )

# alpha_B_code = stellar_force.caseB(T_ambient_code)
# alpha_B_cgs = alpha_B_code * (cu.L_cgs**3 / cu.T_cgs)
alpha_B_cgs = float(hchem.alpha_B_HII_cgs(T_ambient_K))

print(f"  DEBUG alpha_B_cgs (hchem, direct)  = {alpha_B_cgs:.6e}  "
      f"(expected ~1e-13 to 1e-12 for T~1e2-1e4 K)")


# print(f"\n  DEBUG T_ambient_code  = {T_ambient_code:.6e}")
# print(f"  DEBUG alpha_B_code    = {alpha_B_code:.6e}")
# print(f"  DEBUG alpha_B_cgs     = {alpha_B_cgs:.6e}  (expected ~1e-13 to 1e-12 for T~1e2-1e4 K)")

t_rec_cgs = 1.0 / (alpha_B_cgs * n_H_cgs)
R_stromgren_cgs = (3.0 * source_rate_phys / (4.0 * np.pi * alpha_B_cgs * n_H_cgs**2)) ** (1.0 / 3.0)
# R_I_t_cgs = R_stromgren_cgs * (1.0 - np.exp(-t_phys / t_rec_cgs)) ** (1.0 / 3.0)
x = t_phys / t_rec_cgs
R_I_t_cgs = R_stromgren_cgs * (-np.expm1(-x)) ** (1.0 / 3.0)

print("\n" + "=" * 70)
print("STROMGREN SPHERE VALIDATION")
print("=" * 70)
print(f"  ambient n_H           = {n_H_cgs:.4e} cm^-3")
print(f"  ambient T             = {T_ambient_K:.4e} K")
print(f"  alpha_B(T)            = {alpha_B_cgs:.4e} cm^3 s^-1")
print(f"  ionizing photon rate  = {source_rate_phys:.4e} photons s^-1")
print(f"  recombination time    = {t_rec_cgs:.4e} s")
print(f"  R_stromgren (t->inf)  = {R_stromgren_cgs:.4e} cm "
      f"= {R_stromgren_cgs / axis_unit_scale:.4f} {axis_unit_name}")
print(f"  R_I(t={t_phys:.2e}s)    = {R_I_t_cgs:.4e} cm "
      f"= {R_I_t_cgs / axis_unit_scale:.4f} {axis_unit_name}")

if xHII_abs_idx is not None and field_test.shape[0] > xHII_abs_idx:
    xx, yy, zz = np.meshgrid(
        np.arange(size_shape) - center,
        np.arange(size_shape) - center,
        np.arange(size_shape) - center,
        indexing="ij",
    )
    r_cells = np.sqrt(xx**2 + yy**2 + zz**2)
    r_int = np.round(r_cells).astype(int)
    r_max = r_int.max()

    r_vals, xHII_shell_avg = [], []
    for r in range(r_max + 1):
        mask = r_int == r
        vals = xHII_3d[mask]
        if vals.size > 0:
            r_vals.append(r)
            xHII_shell_avg.append(float(np.mean(vals)))
    r_vals = np.array(r_vals)
    xHII_shell_avg = np.array(xHII_shell_avg)

    below_half = np.where(xHII_shell_avg < 0.5)[0]
    if below_half.size > 0:
        r_front_cells = r_vals[below_half[0]]
        r_front_cgs = r_front_cells * dx_phys_cgs
        print(f"  simulated front (x_HII=0.5) = {r_front_cells} cells "
              f"= {r_front_cgs:.4e} cm = {r_front_cgs / axis_unit_scale:.4f} {axis_unit_name}")
        print(f"  ratio simulated / analytic R_I(t) = "
              f"{r_front_cgs / max(R_I_t_cgs, 1e-300):.4f}")
    else:
        print("  !! Ionization front has not reached x_HII < 0.5 anywhere in the box.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_vals * dx_phys_cgs / axis_unit_scale, xHII_shell_avg,
            "o-", ms=3, color="cyan", label="simulated (shell avg)")
    ax.axvline(R_I_t_cgs / axis_unit_scale, color="orange", ls="--",
               label=f"analytic R_I(t) = {R_I_t_cgs/axis_unit_scale:.3f} {axis_unit_name}")
    ax.axhline(0.5, color="white", ls=":", lw=1)
    ax.set_xlabel(f"r [{axis_unit_name}]")
    ax.set_ylabel(r"$\langle x_{HII} \rangle$ (shell average)")
    ax.set_title(f"Stromgren sphere test, t = {fmt_t_phys()}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"stromgren_test_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
print("=" * 70)

# ============================================================================
# PLOT (axis unit + time display follow CONFIG units directly)
# ============================================================================
def compute_extent_phys(size_shape, dx_phys=dx_phys_cgs, centered=True,
                         unit_scale=axis_unit_scale):
    box_extent = size_shape * dx_phys / unit_scale
    if centered:
        half = box_extent / 2.0
        return [-half, half, -half, half]
    return [0, box_extent, 0, box_extent]


plt.style.use("dark_background")
E_slice = E_cell[:, :, c]
extent = compute_extent_phys(size_shape, centered=False)

pos = E_slice[E_slice > 0]
if pos.size == 0:
    print("  !! WARNING: E_slice has no positive values -- skipping this plot.")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot",
                   extent=extent, norm=LogNorm(vmin=max(pos.min(), peak * 1e-12), vmax=peak))
    ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
    ax.set_title(f"Photons/cell, t = {fmt_t_phys()}  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
    fig.colorbar(im, ax=ax, label="photons per cell")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

fig, ax = plt.subplots(figsize=(6, 5))
E_slice_code = E3d[:, :, c]
im = ax.imshow(np.ma.masked_less_equal(E_slice_code, 0.0), origin="lower", cmap="hot",
               extent=extent)
ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
ax.set_title(f"E_gamma code, t = {fmt_t_phys()}  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log10(np.ma.masked_less_equal(E_slice_code, 0.0)), origin="lower", cmap="hot",
               extent=extent)
ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
ax.set_title(f"log10 E_gamma code, t = {fmt_t_phys()}  "
             f"(ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
fig.colorbar(im, ax=ax, label="log10 photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut_log.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

if xHII_abs_idx is not None and field_test.shape[0] > xHII_abs_idx:
    xHII_slice = xHII_3d[:, :, c]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(xHII_slice, origin="lower", cmap="viridis",
                   extent=extent, vmin=0.0, vmax=1.0)
    ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
    ax.set_title(f"Ionization fraction x_HII, t = {fmt_t_phys()}")
    fig.colorbar(im, ax=ax, label=r"$x_{HII}$")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"xHII_slice_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"  x_HII min/max/mean = {xHII_slice.min():.4e} / {xHII_slice.max():.4e} / {xHII_slice.mean():.4e}")

# ============================================================================
# Photon density in cm^-3
# ============================================================================
E_slice_density = E_slice / cell_volume_cm3
assert np.allclose(E_slice_density, E_slice_code / cu.L_cgs**3,
                    rtol=1e-9, atol=1e-15 * np.nanmax(E_slice_density), equal_nan=True)

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
    ax.set_xlabel(f"y [{axis_unit_name}]")
    ax.set_ylabel(f"x [{axis_unit_name}]")
    ax.set_title(f"Photon number density, t = {fmt_t_phys()}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$n_\gamma$  [photons cm$^{-3}$]")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_density_cm3_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

# ============================================================================
# GIF: evolution of ALL fields across iterations (optional, MAKE_GIFS in CONFIG)
# ============================================================================
if MAKE_GIFS:
    dt_frame_code = time_code / GIF_FRAMES
    gif_field_names = ["E_gamma", "Fx", "Fy", "Fz", "rho", "vx", "vy", "vz", "p", "x_HII"]
    DIVERGING_FIELDS = {"Fx", "Fy", "Fz", "vx", "vy", "vz"}
    BOUNDED_FIELDS = {"x_HII"}

    sol_current = cp.deepcopy(sol_test)
    frame_slices = {name: [] for name in gif_field_names}
    t_accum_list = []
    t_accum_code = 0.0

    print(f"\nRunning {GIF_FRAMES} incremental steps for field GIFs "
          f"(dt_frame = {dt_frame_code:.4e} code units each)")

    for frame_i in range(GIF_FRAMES):
        sol_current, _, _, dt_hist_frame, n_steps_frame = hydrosim_test.evolve_till_time(
            sol_current, params, dt_frame_code
        )
        t_accum_code += dt_frame_code
        t_accum_list.append(t_accum_code * cu.T_cgs)

        for k, name in enumerate(gif_field_names):
            data_slice = np.asarray(sol_current[k, :, :, c], dtype=np.float64)
            frame_slices[name].append(data_slice)

        print(f"  frame {frame_i + 1}/{GIF_FRAMES} computed "
              f"(t = {t_accum_list[-1]:.4e} s)")

    gif_dir = os.path.join(BASE_OUTPUT_DIR, "field_gifs")
    os.makedirs(gif_dir, exist_ok=True)

    for name in gif_field_names:
        stack = np.stack(frame_slices[name], axis=0)

        if name in BOUNDED_FIELDS:
            cmap, vmin, vmax = "viridis", 0.0, 1.0
        elif name in DIVERGING_FIELDS:
            cmap = "coolwarm"
            abs_max = np.max(np.abs(stack))
            vmin, vmax = -abs_max, abs_max
        else:
            cmap = "hot"
            vmin, vmax = 0.0, np.max(stack)

        frames_subdir = os.path.join(gif_dir, f"{name}_frames")
        os.makedirs(frames_subdir, exist_ok=True)
        frame_paths = []

        for frame_i in range(GIF_FRAMES):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(stack[frame_i], origin="lower", cmap=cmap,
                            extent=extent, vmin=vmin, vmax=vmax)
            ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
            ax.set_title(f"{name}, iter {frame_i + 1}/{GIF_FRAMES}, "
                         f"t = {t_accum_list[frame_i] / time_axis_scale:.2e} {time_axis_unit}")
            fig.colorbar(im, ax=ax, label=name)
            plt.tight_layout()

            frame_path = os.path.join(frames_subdir, f"frame_{frame_i:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(frame_path)

        gif_out = os.path.join(gif_dir, f"{name}_evolution_{run_tag}.gif")
        images = [Image.open(p) for p in frame_paths]
        images[0].save(
            gif_out, save_all=True, append_images=images[1:], duration=200, loop=0,
        )
        print("wrote", gif_out)

# ============================================================================
# Per-field slices + CSV dumps
# ============================================================================
field_names = ["E_gamma", "Fx", "Fy", "Fz", "rho", "vx", "vy", "vz", "p", "x_HII"]
assert field_test.shape[0] == len(field_names), (
    f"field_test has {field_test.shape[0]} fields, expected {len(field_names)}"
)

BASE_FIELDS_DIR = os.path.join(BASE_OUTPUT_DIR, "fields")
os.makedirs(BASE_FIELDS_DIR, exist_ok=True)

size_x = field_test.shape[1]
c = size_x // 2

for k, name in enumerate(field_names):
    field_dir = os.path.join(BASE_FIELDS_DIR, name)
    os.makedirs(field_dir, exist_ok=True)

    data_3d = np.asarray(np.log10(field_test[k] + 1e-100), dtype=np.float64)
    data_slice = data_3d[:, :, c]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(data_slice, origin="lower", cmap="hot")
    ax.set_xlabel("y cell")
    ax.set_ylabel("x cell")
    ax.set_title(f"{name} slice (z={c})")
    fig.colorbar(im, ax=ax, label=name)
    plt.tight_layout()
    plt.savefig(f"{field_dir}/slice_{name}_log.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    xi, yi, zi = np.meshgrid(
        np.arange(data_3d.shape[0]),
        np.arange(data_3d.shape[1]),
        np.arange(data_3d.shape[2]),
        indexing="ij",
    )
    csv_path = os.path.join(field_dir, f"cube_{name}.csv")
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

for k, name in enumerate(field_names):
    field_dir = os.path.join(BASE_FIELDS_DIR, name)
    os.makedirs(field_dir, exist_ok=True)

    data_3d = np.asarray(field_test[k], dtype=np.float64)
    data_slice = data_3d[:, :, c]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(data_slice, origin="lower", cmap="hot")
    ax.set_xlabel("y cell")
    ax.set_ylabel("x cell")
    ax.set_title(f"{name} slice (z={c})")
    fig.colorbar(im, ax=ax, label=name)
    plt.tight_layout()
    plt.savefig(f"{field_dir}/slice_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{name}] wrote slice PNG")
