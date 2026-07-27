
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
matplotlib.use("Agg") #afficher ou non tous les plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import diffhydro as dh
from diffhydro.units import CodeUnits
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer import StellarRadiationForce

print("Backend:", jax.default_backend(), jax.devices())

# ============================================================================
# PHYSICAL SETUP
# ============================================================================

size_shape        = int(os.environ.get("N", 256))
box_width_phys    = 3.2                        # cm
dx_phys           = box_width_phys / size_shape  # cm per cell
source_rate_phys  = 1e10                         # photons / s

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

# ============================================================================
# RUN TAG: encodes every physical input parameter, used for BOTH the output
# folder name and every saved filename, so every run is self-describing and
# nothing gets overwritten by a different parameter combination.
# ============================================================================
run_tag = (
    f"N{size_shape}"
    f"_box{box_width_phys:.2e}cm"
    f"_v{cu.V_cgs:.2e}cms"
    f"_src{source_rate_phys:.2e}phs"
    f"_t{t_phys:.2e}s"
)

BASE_OUTPUT_DIR = os.path.join("examples/athena/Images_athena", run_tag)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(f"  run_tag               = {run_tag}")
print(f"  output dir            = {BASE_OUTPUT_DIR}")
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
solver_test = dh.LaxFriedrichs_Radiative_transfer(
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
def compute_extent_phys(size_shape, box_width_phys=box_width_phys):
    """
    Convertit les indices de pixels [0, size_shape] en unites physiques,
    pixel 0 -> 0, dernier pixel -> size_shape * cell_size_phys.
    Aucun centrage sur zero.
    """
    box_extent = size_shape * (box_width_phys / size_shape)
    return [0, box_extent, 0, box_extent]


plt.style.use("dark_background")
E_slice = E3d[:, :, c]
extent  = compute_extent_phys(size_shape, box_width_phys)
extent  = [-box_width_phys / 2, box_width_phys / 2] * 2
fig, ax = plt.subplots(figsize=(6, 5))
pos = E_slice[E_slice > 0]
im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot",
               extent=extent, norm=LogNorm(vmin=max(pos.min(), peak * 1e-12), vmax=peak))
ax.set_xlabel("y [cm]"); ax.set_ylabel("x [cm]")
ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per cell")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

fig, ax = plt.subplots(figsize=(6, 5))
pos = E_slice[E_slice > 0]
im = ax.imshow(np.ma.masked_less_equal(E_slice, 0.0), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per cell")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

E_slice = E3d[:, :, c]
extent  = compute_extent_phys(size_shape, box_width_phys)
extent  = [-box_width_phys / 2, box_width_phys / 2] * 2
fig, ax = plt.subplots(figsize=(6, 5))
pos = E_slice[E_slice > 0]
im = ax.imshow(np.log10(np.ma.masked_less_equal(E_slice, 0.0)), origin="lower", cmap="hot",)
ax.set_xlabel("y [cm]"); ax.set_ylabel("x [cm]")
ax.set_title(f"Photons/cell, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per cell")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut_log.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("wrote", out)
# ============================================================================
# NOUVELLE FIGURE : densite de photons en cm^-3 dans la colorbar
# ============================================================================
# Conversion "photons par cellule" -> "photons par cm^3" :
#   n_gamma [cm^-3] = N_gamma [photons/cellule] / dx_phys^3 [cm^3]
# dx_phys est la taille physique d'une cellule (deja calculee plus haut).
cell_volume_cm3 = dx_phys**3
E_slice_density = E_slice / cell_volume_cm3   # photons / cm^3

pos_density = E_slice_density[E_slice_density > 0]
peak_density = E_slice_density.max()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    np.ma.masked_less_equal(E_slice_density, 0.0),
    origin="lower", cmap="hot",
    extent=extent,
    norm=LogNorm(vmin=max(pos_density.min(), peak_density * 1e-12), vmax=peak_density),
)
ax.set_xlabel("y [cm]")
ax.set_ylabel("x [cm]")
ax.set_title(f"Photon number density, t = {t_phys:.2e} s  "
             f"(ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$n_\gamma$  [photons cm$^{-3}$]")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_density_cm3_{run_tag}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
# ============================================================================
# SPHERICAL AVERAGE + POWER-LAW (log-log linear) REGRESSION
# ============================================================================

def spherical_average(field_3d, center):
    cx, cy, cz = center
    nx, ny, nz = field_3d.shape
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    z = np.arange(nz) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    r_int = np.round(R).astype(int)
    r_max = int(r_int.max())

    r_vals, avg_vals = [], []
    for r in range(r_max + 1):
        mask = r_int == r
        vals = field_3d[mask]
        finite = vals[np.isfinite(vals) & (vals > 0.0)]
        if finite.size > 0:
            r_vals.append(r)
            avg_vals.append(float(np.mean(finite)))

    return np.array(r_vals, dtype=float), np.array(avg_vals, dtype=float)


def analyze_inverse_r2(field_3d, size_shape, cell_size_phys, tag,
                       radius_truncation=None,
                       output_dir=BASE_OUTPUT_DIR):
    center_idx = size_shape // 2
    sigma = max(1, round(size_shape // 100))
    injection_radius = len(jnp.arange(-3 * sigma, 3 * sigma + 1)) // 2

    if radius_truncation is None:
        radius_truncation = max(injection_radius + 8, size_shape)

    r_sph, y_sph = spherical_average(
        np.array(field_3d, dtype=float),
        center=(center_idx, center_idx, center_idx)
    )

    mask = (r_sph > injection_radius) & (r_sph < radius_truncation)#(r_sph<80) #
    r_valid = r_sph[mask]
    y_valid = y_sph[mask]
    x_valid = center_idx + r_valid

    if r_valid.size < 5:
        print(f"[{tag}] Not enough valid points for 1/r^2 analysis.")
        return None

    log_r = np.log(r_valid)
    log_y = np.log(y_valid)

    def line_model(x, c, b):
        return c - b * x

    popt, pcov = curve_fit(
        line_model,
        log_r,
        log_y,
        p0=[log_y[0], 2.0],
        maxfev=20000
    )

    c_fit, b = float(popt[0]), float(popt[1])
    b_err = float(np.sqrt(pcov[1, 1])) if pcov.size else np.nan

    y_pred = np.exp(line_model(log_r, c_fit, b))

    print(f"[{tag}] c={c_fit:.6f}  b={b:.6f} (+/- {b_err:.2e})")
    print(f"[{tag}] injection_radius={injection_radius}, fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells")

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.plot(log_r, log_y, "o", color="white", ms=3, label="Spherical avg")
    ax.plot(log_r, np.log(y_pred), "r-", lw=2, label=f"fit a*r^(-b), b={b:.3f}")
    ax.axvline(np.log(injection_radius), color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
    ax.set_xlabel("log(r [cell])")
    ax.set_ylabel("log(Shell-averaged field)")
    ax.set_title(f"log-log spherical average - {tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loglog_spherical_average_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.plot(x_valid, y_valid, "o", color="0.75", ms=3, label="Spherical avg data")
    ax.plot(x_valid, y_pred, "r-", lw=2, label=f"fit a*r^(-b), b={b:.3f}")
    ax.axvline(center_idx + injection_radius, color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
    ax.set_xlabel("Cell index (center + r)")
    ax.set_ylabel("Shell-averaged field value")
    ax.set_title(f"Linear spherical average - {tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/linear_spherical_average_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "tag": tag,
        "r_sph": r_sph,
        "y_sph": y_sph,
        "r_valid": r_valid,
        "y_valid": y_valid,
        "x_valid": x_valid,
        "y_pred": y_pred,
        "c": c_fit,
        "b": b,
        "b_err": b_err,
        "injection_radius": injection_radius,
        "cell_size_phys": cell_size_phys,
    }
def value_average_radius(field_3d, size_shape, cell_size_phys, tag,
                       radius_truncation=None,
                       output_dir=BASE_OUTPUT_DIR):
    center_idx = size_shape // 2
    sigma = max(1, round(size_shape // 100))
    injection_radius = len(jnp.arange(-3 * sigma, 3 * sigma + 1)) // 2

    if radius_truncation is None:
        radius_truncation = max(injection_radius + 8, size_shape)

    r_sph, y_sph = spherical_average(
        np.array(field_3d, dtype=float),
        center=(center_idx, center_idx, center_idx)
    )

    mask = (r_sph > injection_radius) & (r_sph < radius_truncation)#(r_sph<80) #
    r_valid = r_sph[mask]
    y_valid = y_sph[mask]
    x_valid = center_idx + r_valid

    if r_valid.size < 5:
        print(f"[{tag}] Not enough valid points for 1/r^2 analysis.")
        return None

    log_r = np.log(r_valid)
    log_y = np.log(y_valid)

    print(f"[{tag}] injection_radius={injection_radius}, fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells")

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.semilogx(r_valid, log_y, "o", color="white", ms=3, label="Spherical avg")
    ax.axvline(np.log(injection_radius), color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
    ax.set_xlabel("log(r [cell])")
    ax.set_ylabel("log(Shell-averaged field)")
    ax.set_title(f"log-log spherical average - {tag}")
    ax.legend(fontsize=8)
    ax.grid(which="both", color="0.25", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loglog_spherical_average_{tag}_brut.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.plot(x_valid, y_valid, "o", color="0.75", ms=3, label="Spherical avg data")
    ax.axvline(center_idx + injection_radius, color="cyan", ls="--", lw=1, label=f"Injection = {injection_radius}")
    ax.set_xlabel("Cell index (center + r)")
    ax.set_ylabel("Shell-averaged field value")
    ax.set_title(f"Linear spherical average - {tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/linear_spherical_average_{tag}_brut.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "tag": tag,
        "r_sph": r_sph,
        "y_sph": y_sph,
        "r_valid": r_valid,
        "y_valid": y_valid,
        "x_valid": x_valid,
        "injection_radius": injection_radius,
        "cell_size_phys": cell_size_phys,
    }

# tag = run_tag: every saved figure name now embeds all physical parameters
# (grid size, box size, light-speed unit, source rate, physical time).
evolve_value_radius_result = value_average_radius(
    field_3d=E3d,
    size_shape=size_shape,
    cell_size_phys=dx_phys,
    tag=run_tag,
    radius_truncation=157,
    output_dir=BASE_OUTPUT_DIR,
)
fit_result = analyze_inverse_r2(
    field_3d=E3d,
    size_shape=size_shape,
    cell_size_phys=dx_phys,
    tag=run_tag,
    radius_truncation=80,
    output_dir=BASE_OUTPUT_DIR,
)

if fit_result is not None:
    print(f"\nPower-law fit result: n_gamma(r) ~ r^(-{fit_result['b']:.3f})  "
          f"(+/- {fit_result['b_err']:.2e})")
else:
    print("\nPower-law fit could not be performed (not enough valid points).")
