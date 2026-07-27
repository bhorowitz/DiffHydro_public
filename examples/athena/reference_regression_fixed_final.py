"""
RAMSES-RT point-source test, with units that match what the solver actually does.


UNIT CONVENTION (new)
--------------------------------
We no longer derive the length unit from the box:


    old :   unit_length   = box_width_phys / N      (1 cell = 1 code unit,
                                                        dx_code == 1 enforced)
    new:   box_width_phys = box_width_code * unit_length
               dx_code        = box_width_code / N     (arbitrary dx_code)


The two FREE inputs are now
  * unit_length_phys : physical size (cm) of ONE code length unit,
  * box_width_code   : box size expressed in code length units,
and both the physical box size and the cell size follow from them.


Consequence: dx_code is no longer equal to 1, so it must be passed explicitly
to EVERYTHING that contains a dx, otherwise the solver falls back to its default dx_o = 1:


  * hydro(dx=dx_code)                        -> divergence des flux, rhs/dx_o
  * ConvectiveFlux_Radiative_transfer(dx=)   -> CFL, dt = cfl / (ndim*c/dx)
  * StellarRadiationForce(dx=dx_code)        -> source cell volume


And the sol[0] field (E_gamma) is a photon DENSITY in code units
(photons per code-volume unit), no longer "photons per cell":


    photons per cell = E_code * dx_code**3
    n_gamma [cm^-3]     = E_code / cu.L_cgs**3      (= photons_par_cellule / dx_phys**3)
    photons in the box = sum(E_code) * dx_code**3


All the diagnostics/plotting below work with E_cell (photons per
cell), which is invariant under a change of convention: a run
(unit_length = dx_phys, box_width_code = N) redonne exactement l'old cas
dx_code = 1.


VALIDATION (N=64, box = 3.2 cm, dx_phys = 0.05 cm, t = 5.2e-11 s)
-----------------------------------------------------------------
Same physics, four different unit systems:


  ULEN [cm]  BOXCODE  dx_code  steps  t atteint [s]  photons  pic [ph/cell]  b
  0.05        64       1.0      234   5.2036e-11     0.52036  1.1333e-3      2.192
  1.0          3.2     0.05     234   5.2036e-11     0.52036  1.1317e-3      2.170
  0.1         32       0.5      234   5.2036e-11     0.52036  1.1319e-3      2.200
  0.005      640      10.0      234   5.2036e-11     0.52036  1.0180e-3      2.168


And at N=256 (box 3.2 cm, dx_phys = 0.0125 cm), the two extreme conventions:


  ULEN [cm]  BOXCODE  dx_code  steps  photons    pic [ph/cell]  front  b
  0.0125     256      1.0      936    0.5203602  7.5735e-5      127    1.983 +/- 0.004
  1.0          3.2    0.0125   936    0.5203601  7.3868e-5      125    2.004 +/- 0.002
  (expected front = 124.7 cellules)


La ligne dx_code = 1 reproduit exactement l'old script (verified a N=64 :
same E min/max, same sum(dt), same front radius).  The number of steps, the reached physical time, the
front radius (31 cells for 31.2 expected) and the total photons are
strictly invariant; only the profile AMPLITUDE changes slightly (0.1% for
dx_code entre 0.05 et 1, 10 % pour dx_code = 10).  This drift follows only
the field amplitude IN CODE UNITS (E_code = ph/cellule / dx_code**3, verified
by rerunning dx_code=10 with SRC x1e3: the peak returns to 1.1333) : ce sont
les seuils absolus du solveur (eq.eps, les "+1e-30") et le float32, pas le
dx plumbing.  Hence the practical rule: choose unit_length so that E_code
stays well above eq.eps (the script prints the diagnostic).


Environment variables : GPU, N, ULEN, BOXCODE, UVEL, SRC, TPHYS, EPS, MAXDT,
NSTEP, RTRUNC_AVG, RTRUNC_FIT.


Pour retrouver exactement l'oldne convention (dx_code = 1) :
    N=256 ULEN=0.0125 BOXCODE=256 python examples/athena/reference_regression_fixed_final.py
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
matplotlib.use("Agg") # whether or not to display all plots
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


size_shape         = int(os.environ.get("N", 256))
# --- free inputs of the new convention ------------------------------
unit_length_phys   = float(os.environ.get("ULEN", 1.0))      # cm  <- 1 code length unit
box_width_phys     = float(os.environ.get("BOXPHYS", 10))   # physical length units cm
# box_width_code     = float(os.environ.get("BOXCODE", 3.2))   # code length units
unit_velocity_phys = float(os.environ.get("UVEL", 3e10))     # cm/s <- 1 code velocity unit


# --- everything else is derived ---------------------------------------------
# box_width_phys     = box_width_code * unit_length_phys       # cm
box_width_code     = box_width_phys / unit_length_phys       # code length units
dx_code            = box_width_code / size_shape             # code units per cell
dx_phys            = dx_code * unit_length_phys              # cm per cell
cell_volume_code   = dx_code ** 3                            # cell volume, code units
cell_volume_cm3    = dx_phys ** 3                            # cell volume, cm^3


source_rate_phys   = float(os.environ.get("SRC", 1e10))      # photons / s


# ct must stay inside the (periodic) box: ct < box/2  ->  t < box/(2c).
# The original t = 5.2e-11 s gives ct = 1.56 cm = 0.49 box widths for box = 3.2 cm.
t_phys = float(os.environ.get("TPHYS", 5.2e-11))   # s


# --- code units: 1 code length unit = unit_length_phys cm -----------
cu = CodeUnits.from_config(
    {"length": f"{unit_length_phys} cm",
     "mass": "1 g",
     "velocity": f"{unit_velocity_phys} cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)


c_cgs             = 2.99792458e10
light_speed_code  = c_cgs / cu.V_cgs          # ~1.0
time_code         = t_phys / cu.T_cgs
source_rate_code  = source_rate_phys * cu.T_cgs   # photons per code time


# CFL of the RT flux: dt = cfl / (ndim * c / dx)  -> now depends on dx_code.
cfl_code   = 0.4                                  # = EquationManager_RT.cfl
dt_cfl     = cfl_code / (3.0 * light_speed_code / dx_code)
n_steps_est = int(math.ceil(time_code / dt_cfl))
# max_dt must not override the CFL limit (hydro default = 0.5, which caps large dx_code values).
max_dt      = float(os.environ.get("MAXDT", 2.0 * dt_cfl))
n_super_step = int(os.environ.get("NSTEP", int(1.2 * n_steps_est) + 100))


# ============================================================================
# RUN TAG: encodes every physical input parameter, used for BOTH the output
# folder name and every saved filename, so every run is self-describing and
# nothing gets overwritten by a different parameter combination.
# ============================================================================
run_tag = (
    f"N{size_shape}"
    f"_ulen{unit_length_phys:.2e}cm"
    f"_boxc{box_width_code:.2e}"
    f"_box{box_width_phys:.2e}cm"
    f"_v{cu.V_cgs:.2e}cms"
    f"_src{source_rate_phys:.2e}phs"
    f"_t{t_phys:.2e}s"
)


BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/athena/Images_athena", run_tag)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print(f"  run_tag               = {run_tag}")
print(f"  output dir            = {BASE_OUTPUT_DIR}")
print("=" * 70)
print("  --- convention box_size = box_width_code * unit_length ---")
print(f"  unit_length_phys      = {unit_length_phys:.6e} cm   (1 code length unit)")
print(f"  box_width_code        = {box_width_code:.6e} unites code")
print(f"  box_width_phys        = {box_width_phys:.6e} cm")
print(f"  dx_code               = {dx_code:.6e} unites code / cellule")
print(f"  dx_phys               = {dx_phys:.6e} cm / cellule")
print("  --- code -> cgs scales ---")
print(f"  L_cgs                 = {cu.L_cgs:.6e} cm")
print(f"  V_cgs                 = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs                 = {cu.T_cgs:.6e} s")
print(f"  light_speed_code      = {light_speed_code:.6f}")
print(f"  time_code             = {time_code:.6e}")
print(f"  source_rate_code      = {source_rate_code:.4e} photons / code time")
print("  --- time step ---")
print(f"  dt_cfl (expected)      = {dt_cfl:.6e} code = {dt_cfl * cu.T_cgs:.4e} s")
print(f"  max_dt                = {max_dt:.6e} code")
print(f"  estimated n_steps        = {n_steps_est}   (n_super_step = {n_super_step})")
print(f"  expected front  c*t   = {c_cgs * t_phys:.4e} cm "
      f"= {c_cgs * t_phys / dx_phys:.1f} cells "
      f"= {c_cgs * t_phys / box_width_phys:.3f} box widths")
print("=" * 70)


# ============================================================================
# SOLVER
# ============================================================================


# eq.eps is an ABSOLUTE floor in code units (jnp.maximum(E, eps) in
# get_conservatives_from_primitives).  Since E_code = photons_per_cell /
# dx_code**3, changing unit_length changes the field amplitude and therefore the part
# of the profile that gets clipped by eps: eps must stay << typical E_code.
eps_code = float(os.environ.get("EPS", 1e-10))
eq_test = EquationManager_RT(
    light_speed=light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    eps=eps_code,
    debug=False,
)
assert abs(cfl_code - eq_test.cfl) < 1e-12, (
    f"desynchronized cfl: dt_cfl calcule avec {cfl_code}, solver has {eq_test.cfl}"
)
# ordre de grandeur de la densite injectee par time step, a comparer a eps
source_density_per_step = source_rate_code * dt_cfl / cell_volume_code
print(f"  eps_code              = {eps_code:.3e}   "
      f"(source/step = {source_density_per_step:.3e} [ph/vol code], "
      f"ratio = {source_density_per_step / eps_code:.2e})")
if source_density_per_step < 1e4 * eps_code or source_density_per_step < 1e-5:
    print("  !! WARNING: the field amplitude in code units is low. "
          "The solver's absolute thresholds (eps, the +1e-30 values) and float32 start to affect "
          "the profile: measured at N=64, a field around ~1e-6 in code units loses "
          "~10% sur le pic (the total photons remain exact). "
          "Use a larger unit_length (smaller dx_code) or lower EPS.")
solver_test = dh.LaxFriedrichs_Radiative_transfer(
    equation_manager=eq_test, signal_speed=dh.signal_speed_Rusanov
)
# dx_code must be provided here: it sets the CFL of the RT flux.
cf_test = dh.ConvectiveFlux_Radiative_transfer(
    eq_test, solver_test, dh.PLM(limiter="VANLEER"), dx=dx_code
)


stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=dx_code,                    # cell volume: rate [ph/t] -> density [ph/vol]
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=False,
    gaussian_star=True,            # gaussian_star=False breaks under jit (python `if` on tracer)
    injection_geometry="3D",
    eq=eq_test,
    debug=False,
    momentum_only=False,
)


hydrosim_test = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[cf_test],
    forces=[stellar_force],
    dx=dx_code,                    # flux divergence: rhs / dx_o
    max_dt=max_dt,
)
assert hydrosim_test.dx_o == cf_test.dx_o == stellar_force.dx, "desynchronized dx !"
print("hydrosim_test.dx_o =", hydrosim_test.dx_o, " cf.dx_o =", cf_test.dx_o,
      " force.dx =", stellar_force.dx, " cfl =", eq_test.cfl)
print("expected dt_code   =", eq_test.cfl / (3.0 * light_speed_code / dx_code))


params = {
    "star_masses":        jnp.array([1.0]),
    "star_ages":          jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions":     jnp.array([[size_shape // 2] * 3], dtype=jnp.int32),
}
sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))


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
    print(f"  !! ATTENTION: n_steps saturated n_super_step={n_super_step}: "
          f"t_target is NOT reached. Increase NSTEP.")
if dt_hist[0] < 0.99 * dt_cfl:
    print(f"  !! ATTENTION: dt ({dt_hist[0]:.3e}) < dt_cfl ({dt_cfl:.3e}): "
          f"max_dt={max_dt:.3e} caps the CFL.")


# ============================================================================
# DIAGNOSTICS
# ============================================================================
# E3d      : code field, photon DENSITY [photons / code volume]
# E_cell   : photons per cell (convention invariant) = E3d * dx_code**3
# E_dens   : physical density [photons / cm^3]             = E_cell / dx_phys**3


# float64 for diagnostics: the solver field is in float32, and multiplying
# par dx_code**3 (1.95e-6 a N=256) would underflow the tail of the profile.
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
      f"= {source_rate_phys * dt_sum * cu.T_cgs:.6e} ph expecteds (cgs)")


line = E_cell[c:, c, c]
peak = E_cell.max()
for th in [1e-3, 1e-6, 1e-10, 1e-15]:
    idx = np.where(line > peak * th)[0]
    r = idx.max() if idx.size else 0
    print(f"  thr {th:.0e} of peak -> radius {r:3d} cells = {r * dx_phys:.4e} cm "
          f"= {r * dx_code:.4e} code units")
print(f"  expected free-streaming radius = {c_cgs * t_phys / dx_phys:.1f} cells")


# ============================================================================
# PLOT
# ============================================================================
def compute_extent_phys(size_shape, dx_phys=dx_phys, centered=True):
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
extent  = compute_extent_phys(size_shape)
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


# raw solver field = density in code units, axes in cell indices
fig, ax = plt.subplots(figsize=(6, 5))
E_slice_code = E3d[:, :, c]
pos_code = E_slice_code[E_slice_code > 0]
im = ax.imshow(np.ma.masked_less_equal(E_slice_code, 0.0), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"E_gamma code, t = {t_phys:.2e} s  (ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)


fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log10(np.ma.masked_less_equal(E_slice_code, 0.0)), origin="lower", cmap="hot")
ax.set_xlabel("y cell"); ax.set_ylabel("x cell")
ax.set_title(f"log10 E_gamma code, t = {t_phys:.2e} s  "
             f"(ct = {c_cgs*t_phys/dx_phys:.0f} cells)")
fig.colorbar(im, ax=ax, label="log10 photons per code volume")
plt.tight_layout()
out = os.path.join(BASE_OUTPUT_DIR, f"field_test_fixed_units_{run_tag}_brut_log.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
# ============================================================================
# NEW FIGURE: photon density in cm^-3 in the colorbar
# ============================================================================
# Conversion "photons per cell" -> "photons par cm^3" :
#   n_gamma [cm^-3] = N_gamma [photons/cellule] / dx_phys^3 [cm^3]
#                   = E_code / L_cgs^3          (the two routes agree)
E_slice_density = E_slice / cell_volume_cm3   # photons / cm^3
# the two conversion routes must agree (up to machine precision)
assert np.allclose(E_slice_density, E_slice_code / cu.L_cgs**3,
                   rtol=1e-9, atol=1e-15 * E_slice_density.max())


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
# r is in CELLS (indices). r_phys = r * dx_phys [cm], r_code = r * dx_code.
# The fitted exponent b is invariant under a change of units (constant factor).


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
    print(f"[{tag}] injection_radius={injection_radius}, "
          f"fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells "
          f"= [{r_valid.min()*cell_size_phys:.3e}, {r_valid.max()*cell_size_phys:.3e}] cm")


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


    print(f"[{tag}] injection_radius={injection_radius}, "
          f"fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells")


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
# (grid size, unit length, box size, light-speed unit, source rate, physical time).
# We analyze E_cell (photons/cell): independent of the unit convention.
def clamp_truncation(r_trunc, label):
    """Beyond N/2 the spherical shells leave the periodic box."""
    r_max = size_shape // 2
    if r_trunc > r_max:
        print(f"  [{label}] radius_truncation {r_trunc} > N/2 = {r_max}: "
              f"clamped to {r_max} (shells outside the box).")
        return r_max
    return r_trunc



evolve_value_radius_result = value_average_radius(
    field_3d=E_cell,
    size_shape=size_shape,
    cell_size_phys=dx_phys,
    tag=run_tag,
    radius_truncation=clamp_truncation(int(os.environ.get("RTRUNC_AVG", 157)), "avg"),
    output_dir=BASE_OUTPUT_DIR,
)
fit_result = analyze_inverse_r2(
    field_3d=E_cell,
    size_shape=size_shape,
    cell_size_phys=dx_phys,
    tag=run_tag,
    radius_truncation=clamp_truncation(int(os.environ.get("RTRUNC_FIT", 80)), "fit"),
    output_dir=BASE_OUTPUT_DIR,
)


if fit_result is not None:
    print(f"\nPower-law fit result: n_gamma(r) ~ r^(-{fit_result['b']:.3f})  "
          f"(+/- {fit_result['b_err']:.2e})")
else:
    print("\nPower-law fit could not be performed (not enough valid points).")
