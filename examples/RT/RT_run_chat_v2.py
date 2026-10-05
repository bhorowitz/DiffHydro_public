"""
RAMSES-RT point-source test, with units that match what the solver actually does.

UNIT CONVENTION (unit-agnostic via UnitParser)
----------------------------------------------------------
Every physical length/velocity/time input can be given in ANY unit known to
diffhydro.units.registry.UnitParser. Each input is parsed once, converted to
cgs immediately, and everything downstream (dx_code, CodeUnits, the solver,
the plots) only ever sees cgs values.

AXIS DISPLAY UNIT: every figure's axis label/title auto-adapts its unit
(cm, m, km, pc, kpc, ...) to the physical box size, via auto_length_unit(),
which reads the SAME UnitParser table used to parse ULEN/BOXPHYS. Changing
N, ULEN or BOXPHYS changes both the output folder name (run_tag) and the
axis unit shown on every plot -- no manual edits needed.

FIELD LAYOUT (depends on the CURRENT EquationManager, which now treats
x_HII as a generic named PASSIVE SCALAR rather than a hard-coded slot):

  eq_test_hydro = dh.EquationManager(gamma=..., n_cons=6,
                                      passive_names=("x_HII",), ...)

  -> within the hydro block, order is ACTIVE first, PASSIVE after:
       rho, vx, vy, vz, p, x_HII

  -> combined sol layout (RT block has n_cons=4: E,Fx,Fy,Fz):
       0: E_gamma  1: Fx  2: Fy  3: Fz
       4: rho      5: vx  6: vy  7: vz  8: p
       9: x_HII

This is DIFFERENT from an older convention where x_HII sat right after the
RT block (index 4) and hydro started at index 5. If you ever change
passive_names or n_cons on eq_test_hydro, field_names / sol_test indices
below must be updated to match, since EquationManager itself is NOT
modified here.

Environment variables (ULEN, BOXPHYS, UVEL, TPHYS accept ANY unit string
recognized by UnitParser, e.g. "1 km", "3.2 cm", "3e5 km/s", "5.2e-11 s"):
  GPU, N, ULEN, BOXPHYS, BOXCODE, UVEL, SRC, TPHYS, EPS, MAXDT, NSTEP
"""

import os, sys, math
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
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
from diffhydro.units.registry import UnitParser
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
    unit). Requires UnitParser.units_for_dimension(dim) to exist; falls
    back to plain cm if that method is unavailable."""
    if not hasattr(parser, "units_for_dimension"):
        return "cm", 1.0
    units = parser.units_for_dimension("length")  # sorted small -> large
    if not units:
        return "cm", 1.0
    best = units[0]
    for name, factor in units:
        if value_cgs / factor >= 1.0:
            best = (name, factor)
    return best


# ============================================================================
# PHYSICAL SETUP
# ============================================================================

size_shape = int(os.environ.get("N", 100))

ulen_q = env_quantity("ULEN", "1.0 cm", expected_dim="length")
unit_length_phys_cgs = ulen_q.cgs_value
unit_length_str = f"{ulen_q.value:g} {ulen_q.unit}"

uvel_q = env_quantity("UVEL", "3e10 cm/s", expected_dim="velocity")
unit_velocity_phys = uvel_q.cgs_value

if "BOXPHYS" in os.environ:
    boxphys_q = env_quantity("BOXPHYS", "3.2 cm", expected_dim="length")
    box_width_phys_cgs = boxphys_q.cgs_value
    box_width_code = box_width_phys_cgs / unit_length_phys_cgs
    box_width_str = f"{boxphys_q.value:g} {boxphys_q.unit}"
else:
    box_width_code = float(os.environ.get("BOXCODE", 3.2))
    box_width_phys_cgs = box_width_code * unit_length_phys_cgs
    box_width_str = f"{box_width_phys_cgs:.3e} cm"

dx_code = box_width_code / size_shape
dx_phys_cgs = dx_code * unit_length_phys_cgs
cell_volume_code = dx_code ** 3
cell_volume_cm3 = dx_phys_cgs ** 3

source_rate_phys = float(os.environ.get("SRC", 1e10))

tphys_q = env_quantity("TPHYS", "5.2e-11 s", expected_dim="time")
t_phys = tphys_q.cgs_value

cu = CodeUnits.from_config(
    {"length": f"{unit_length_phys_cgs} cm",
     "mass": "1 g",
     "velocity": f"{unit_velocity_phys} cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

c_cgs = 2.99792458e10
light_speed_code = c_cgs / cu.V_cgs
time_code = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs

cfl_code = 0.4
dt_cfl = cfl_code / (3.0 * light_speed_code / dx_code)
n_steps_est = int(math.ceil(time_code / dt_cfl))
max_dt = float(os.environ.get("MAXDT", 2.0 * dt_cfl))
n_super_step = int(os.environ.get("NSTEP", int(1.2 * n_steps_est) + 100))

axis_unit_name, axis_unit_scale = auto_length_unit(box_width_phys_cgs)

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
print(f"  ULEN input            = '{unit_length_str}'  -> {unit_length_phys_cgs:.6e} cm")
print(f"  BOX  input            = '{box_width_str}'    -> {box_width_phys_cgs:.6e} cm")
print(f"  UVEL input            = '{uvel_q.value:g} {uvel_q.unit}' -> {unit_velocity_phys:.6e} cm/s")
print(f"  TPHYS input           = '{tphys_q.value:g} {tphys_q.unit}' -> {t_phys:.6e} s")
print(f"  dx_code               = {dx_code:.6e} code units / cell")
print(f"  dx_phys_cgs           = {dx_phys_cgs:.6e} cm / cell")
print(f"  L_cgs                 = {cu.L_cgs:.6e} cm")
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

eps_code = float(os.environ.get("EPS", 1e-20))
eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    debug=False,
)

# --- ONLY change needed for the "Unknown passive scalar 'x_HII'" error ---
# EquationManager treats x_HII as a generic named passive scalar: it must
# be declared via passive_names, and n_cons must be big enough to hold it
# (5 active: rho,vx,vy,vz,p + 1 passive: x_HII = 6). We do NOT touch
# equationmanager.py: this is purely how we instantiate it here.
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

params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}

# --- Combined layout with n_cons=6 on the hydro block (rho,vx,vy,vz,p,x_HII):
#   0: E_gamma  1: Fx  2: Fy  3: Fz  4: rho  5: vx  6: vy  7: vz  8: p  9: x_HII
n_total_fields = eq_test.n_cons + eq_test_hydro.n_cons
sol_test = jnp.zeros((n_total_fields, size_shape, size_shape, size_shape), dtype=jnp.float32)
center = size_shape // 2

idx_rho_local = eq_test.n_cons + eq_test_hydro.mass_ids       # 4
idx_p_local = eq_test.n_cons + eq_test_hydro.energy_ids       # 8

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

sol_test = sol_test.at[idx_rho_local].set(rho_ambient_code)
sol_test = sol_test.at[idx_p_local].set(p_ambient_code)
sol_test = sol_test.at[0, center, center, center].set(1e-20)

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

# x_HII diagnostic: use stellar_force.idx_xHII (computed dynamically from
# hydro_eq.xHII_id), NOT eq_test.xHII_id/eq_test.n_active which belong to
# the RT manager and were never correct here.
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
# PLOT  (axis unit auto-adapted via axis_unit_name / axis_unit_scale)
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
    ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys_cgs:.0f} cells)")
    fig.colorbar(im, ax=ax, label="photons per cell")
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

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

if xHII_abs_idx is not None and field_test.shape[0] > xHII_abs_idx:
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
# Per-field slices + CSV dumps
# ============================================================================
# Layout matches n_cons=6 on the hydro block: rho,vx,vy,vz,p (active), then
# x_HII (passive), preceded by the 4 RT fields.
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
