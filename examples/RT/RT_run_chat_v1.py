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

NEW: the axis labels / titles of every figure now auto-adapt their display
unit (cm, m, km, pc, kpc, ...) to the physical box size, using the SAME
UnitParser table (registry.py -> units_for_dimension) that parses ULEN/BOXPHYS.
Changing N, ULEN or BOXPHYS therefore changes both the output folder name
(run_tag, already dynamic) AND the axis unit shown on every plot -- no manual
edits needed.

old : unit_length = box_width_phys / N (1 cell = 1 code unit, dx_code == 1 enforced)
new : box_width_phys_cgs = box_width_code * unit_length_phys_cgs
      dx_code = box_width_code / N (arbitrary dx_code)

Environment variables (ULEN, BOXPHYS, UVEL, TPHYS accept ANY unit string
recognized by UnitParser, e.g. "1 km", "3.2 cm", "3e5 km/s", "5.2e-11 s"):
  GPU, N, ULEN, BOXPHYS, BOXCODE, UVEL, SRC, TPHYS, EPS, MAXDT, NSTEP,
  RTRUNC_AVG, RTRUNC_FIT

If BOXPHYS is set, it takes precedence over BOXCODE. SRC (photons/s) has no
length/mass/time dimension in the unit table, so it stays a plain float
(interpreted as cgs photons/s).
"""

import os, sys, math
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

print("Backend:", jax.default_backend(), jax.devices())

up = UnitParser()


def env_quantity(name: str, default: str, expected_dim: str):
    """Read an env var as a free-form quantity string ('1 km', '3.2 cm',
    '3e10 cm/s', '5.2e-11 s', ...), parse it with UnitParser, and return the
    ParsedQuantity (value, unit, dimension, cgs_value)."""
    text = os.environ.get(name, default)
    try:
        return up.parse(text, expected_dim=expected_dim)
    except ValueError as exc:
        raise SystemExit(f"Invalid value for {name}='{text}': {exc}") from exc


def auto_length_unit(value_cgs, parser: UnitParser = up):
    """Pick the length unit (name, cgs_per_unit) from UnitParser's own
    table that keeps `value_cgs` in a human-readable range (>= 1 in that
    unit), falling back to the smallest known length unit if value_cgs is
    smaller than all of them. This is what makes every axis label / title
    auto-adapt when N, ULEN or BOXPHYS change: nothing here is hardcoded
    to "cm", it always re-derives from the actual box size in cgs.
    """
    units = parser.units_for_dimension("length")  # sorted small -> large
    if not units:
        return "cm", 1.0
    best = units[0]  # smallest unit as fallback
    for name, factor in units:
        if value_cgs / factor >= 1.0:
            best = (name, factor)
    return best


# ============================================================================
# PHYSICAL SETUP  (unit-agnostic: any input unit is converted to cgs here,
# and only cgs values are used from this point on)
# ============================================================================

size_shape = int(os.environ.get("N", 100))

ulen_q = env_quantity("ULEN", "1.0 cm", expected_dim="length")
unit_length_phys_cgs = ulen_q.cgs_value        # cm  <- 1 code length unit
unit_length_str      = f"{ulen_q.value:g} {ulen_q.unit}"

uvel_q = env_quantity("UVEL", "3e10 cm/s", expected_dim="velocity")
unit_velocity_phys = uvel_q.cgs_value          # cm/s <- 1 code velocity unit

if "BOXPHYS" in os.environ:
    boxphys_q = env_quantity("BOXPHYS", "3.2 cm", expected_dim="length")
    box_width_phys_cgs = boxphys_q.cgs_value                 # cm
    box_width_code     = box_width_phys_cgs / unit_length_phys_cgs
    box_width_str      = f"{boxphys_q.value:g} {boxphys_q.unit}"
else:
    box_width_code = float(os.environ.get("BOXCODE", 3.2))  # code length units
    box_width_phys_cgs = box_width_code * unit_length_phys_cgs
    box_width_str = f"{box_width_phys_cgs:.3e} cm"

dx_code          = box_width_code / size_shape          # code units per cell
dx_phys_cgs      = dx_code * unit_length_phys_cgs        # cm per cell
cell_volume_code = dx_code ** 3                          # cell volume, code units
cell_volume_cm3  = dx_phys_cgs ** 3                      # cell volume, cm^3

source_rate_phys = float(os.environ.get("SRC", 1e10))    # photons / s

tphys_q = env_quantity("TPHYS", "5.2e-11 s", expected_dim="time")
t_phys = tphys_q.cgs_value   # s

cu = CodeUnits.from_config(
    {"length": f"{unit_length_phys_cgs} cm",
     "mass": "1 g",
     "velocity": f"{unit_velocity_phys} cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

c_cgs            = 2.99792458e10
light_speed_code = c_cgs / cu.V_cgs
time_code        = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs

cfl_code    = 0.4
dt_cfl      = cfl_code / (3.0 * light_speed_code / dx_code)
n_steps_est = int(math.ceil(time_code / dt_cfl))
max_dt       = float(os.environ.get("MAXDT", 2.0 * dt_cfl))
n_super_step = int(os.environ.get("NSTEP", int(1.2 * n_steps_est) + 100))

# ============================================================================
# AXIS DISPLAY UNIT: auto-selected from the physical box size, using the
# SAME UnitParser table used to parse ULEN/BOXPHYS. Every figure axis
# label/title below reads axis_unit_name/axis_unit_scale instead of a
# hardcoded "cm" -- so it adapts automatically to any box size change.
# ============================================================================
axis_unit_name, axis_unit_scale = auto_length_unit(box_width_phys_cgs)

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
print(f"  axis display unit     = {axis_unit_name} (auto-selected from box size)")
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
    passive_names=("x_HII",),
    n_cons=6,
)
assert abs(cfl_code - eq_test.cfl) < 1e-12, (
    f"desynchronized cfl: dt_cfl computed with {cfl_code}, solver has {eq_test.cfl}"
)
source_density_per_step = source_rate_code * dt_cfl / cell_volume_code
print(f"  eps_code              = {eps_code:.3e}   "
      f"(source/step = {source_density_per_step:.3e} [ph/vol code], "
      f"ratio = {source_density_per_step / eps_code:.2e})")
if source_density_per_step < 1e4 * eps_code or source_density_per_step < 1e-5:
    print("  !! WARNING: the field amplitude in code units is low. "
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

hydrosim_test = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar_force],
    dx=dx_code,
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

sol_test = jnp.zeros((10, size_shape, size_shape, size_shape), dtype=jnp.float32)
center = size_shape // 2

# Ambient neutral hydrogen medium, uniform everywhere (Stromgren sphere setup)
n_H_cgs = 1.0
T_ambient_K = 100.0
rho_ambient_cgs = n_H_cgs * 1.6726219e-24
rho_ambient_code = rho_ambient_cgs / cu.rho_cgs
kB_cgs = 1.380649e-16
mH_cgs = 1.6726219e-24
mu = getattr(cu, "mu", 0.61)
p_ambient_cgs = rho_ambient_cgs * kB_cgs * T_ambient_K / (mu * mH_cgs)
p_ambient_code = p_ambient_cgs / cu.P_cgs

sol_test = sol_test.at[5].set(rho_ambient_code)
sol_test = sol_test.at[9].set(p_ambient_code)
sol_test = sol_test.at[0, center, center, center].set(1e-20)

print(f"\nRunning to t = {t_phys:.3e} s = {time_code:.3e} code units ...")
field_test, _, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test), params, time_code
)

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
E3d    = np.asarray(field_test[0], dtype=np.float64)
E_cell = E3d * cell_volume_code
c      = size_shape // 2

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

if field_test.shape[0] > eq_test.n_active:
    xHII_3d = np.asarray(field_test[eq_test.xHII_id], dtype=np.float64)
    print(f"  x_HII min/max/mean = "
          f"{xHII_3d.min():.4e} / {xHII_3d.max():.4e} / {xHII_3d.mean():.4e}")
    n_out_of_bounds = np.sum((xHII_3d < -1e-9) | (xHII_3d > 1.0 + 1e-9))
    print(f"  x_HII cells out of [0,1] bounds = {n_out_of_bounds}")
else:
    print("  !! x_HII not present in field_test (n_cons/passive_names not configured)")

# ============================================================================
# PLOT  (axis unit auto-adapted via axis_unit_name / axis_unit_scale)
# ============================================================================
def compute_extent_phys(size_shape, dx_phys=dx_phys_cgs, centered=True,
                         unit_scale=axis_unit_scale):
    """
    Converts pixel indices [0, size_shape] into the auto-selected axis
    unit (unit_scale = cgs value of one axis unit): one pixel = dx_phys cm,
    divided by unit_scale to land in axis_unit_name units.
    """
    box_extent = size_shape * dx_phys / unit_scale
    if centered:
        half = box_extent / 2.0
        return [-half, half, -half, half]
    return [0, box_extent, 0, box_extent]


plt.style.use("dark_background")
E_slice = E_cell[:, :, c]
extent  = compute_extent_phys(size_shape, centered=False)

pos = E_slice[E_slice > 0]
if pos.size == 0:
    print("  !! WARNING: E_slice has no positive values -- skipping this plot.")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot",
                   extent=extent, norm=LogNorm(vmin=max(pos.min(), peak * 1e-12), vmax=peak))
    ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
    ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
    fig.colorbar(im, ax=ax, label="photons per cell")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

# raw solver field: density in code units, axes stay in cell indices (no
# physical unit to auto-adapt here since these plots are diagnostic-only).
fig, ax = plt.subplots(figsize=(6, 5))
E_slice_code = E3d[:, :, c]
im = ax.imshow(np.ma.masked_less_equal(E_slice_code, 0.0), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"E_gamma code, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
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
print("wrote", out)

if field_test.shape[0] > eq_test.n_active:
    xHII_slice = xHII_3d[:, :, c]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(xHII_slice, origin="lower", cmap="viridis",
                   extent=extent, vmin=0.0, vmax=1.0)
    ax.set_xlabel(f"y [{axis_unit_name}]"); ax.set_ylabel(f"x [{axis_unit_name}]")
    ax.set_title(f"Ionization fraction x_HII, t = {t_phys:.2e} s")
    fig.colorbar(im, ax=ax, label=r"$x_{HII}$")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"xHII_slice_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

# ============================================================================
# Photon density in cm^-3, axis unit auto-adapted
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
    ax.set_title(f"Photon number density, t = {t_phys:.2e} s")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$n_\gamma$  [photons cm$^{-3}$]")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_density_cm3_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

# ============================================================================
# Per-field slices + CSV dumps (folder name already adapts via run_tag)
# ============================================================================
field_names = ["E_gamma", "Fx", "Fy", "Fz", "fractionX_HII", "rho", "vx", "vy", "vz", "p"]
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

    data_3d = np.asarray(field_test[k], dtype=np.float32)
    data_slice = data_3d[:, :, c]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(data_slice, origin="lower", cmap="hot")
    ax.set_xlabel("y cell")
    ax.set_ylabel("x cell")
    ax.set_title(f"{name} slice (z={c}) - {run_tag}")
    fig.colorbar(im, ax=ax, label=name)
    plt.tight_layout()
    plt.savefig(f"{field_dir}/slice_{name}_{run_tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

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
