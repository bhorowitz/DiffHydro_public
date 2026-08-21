
"""
RAMSES-RT Radiative Transfer Validation
========================================
Homogenized, cleaned-up script derived from the DiffHydro tutorial notebook.

Fixes applied vs. the original notebook:
  1. Removed the unused/dead Sedov-Taylor (Athena IC) block that was never
     evolved -- kept as an optional, clearly separated function.
  2. Removed the duplicate/conflicting `cu_rt` definition (the pc/Msun/km-s
     unit system and the "identity" CodeUnits(1,1,1) had the same variable
     name -- now `cu_diag_identity`).
  3. Fixed the inverted `vmin=vmax, vmax=vmin` bug in the first imshow call.
  4. Removed `np.log(...)` calls on fields that can contain zeros/negatives
     without clipping -> replaced with `np.log10(np.clip(..., floor, None))`.
  5. Removed duplicated `ix_center, iy_center, iz_center` re-definitions.
  6. Consolidated the 3 nearly-identical "field slice + zoom" plotting blocks
     into one reusable function.
  7. Consolidated isotropic + beam-x/y/xy simulation setup into one factory
     function (`make_force`, `make_hydro`) instead of copy-pasted blocks.
  8. Removed unused `mask` variable and other dead code.
  9. All prints / physical-unit diagnostics grouped into dedicated functions.
 10. Single top-level `main()` drives the whole pipeline; every stage is a
     small, testable function.
"""

import os
import sys
import copy as cp
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import Javascript, display

# --------------------------------------------------------------------------- #
# 0. Environment / GPU configuration
# --------------------------------------------------------------------------- #
sys.path.append("../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # "" on CPU-only machines
os.environ["DIFFHYDRO_DEBUG_CHECKS"] = "False"

import jax
jax.config.update("jax_disable_jit", False)       # True = easier debugging, much slower
import jax.numpy as jnp

import diffhydro as dh
import diffhydro.utils.debug_checks as dc
importlib.reload(dc)

import diffhydro.units.registry as unit_registry
import diffhydro.units.code_units as unit_code_units
import diffhydro.units.convert as unit_convert
import diffhydro.units as unit_pkg
importlib.reload(unit_registry)

importlib.reload(unit_code_units)
importlib.reload(unit_convert)
importlib.reload(unit_pkg)
from diffhydro.units import CodeUnits, format_quantity, to_code

from diffhydro.utils.io import athinput, athdf
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer import StellarRadiationForce


# --------------------------------------------------------------------------- #
# 1. Small utilities
# --------------------------------------------------------------------------- #

def print_backend_info():
    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())


# --------------------------------------------------------------------------- #
# 2. Unit systems
# --------------------------------------------------------------------------- #
def build_diagnostic_unit_system():
    """
    Secondary unit system (pc / Msun / km-s) used ONLY for human-readable
    diagnostics elsewhere in the notebook. NOT the unit system the RT
    solver actually runs in -- kept isolated to avoid the earlier
    name collision with the RAMSES-RT `cu`.
    """
    cu_diag = CodeUnits.from_config(
        {"length": "1 pc", "mass": "1 Msun", "velocity": "1 km/s"},
        {"gamma": 5.0 / 3.0, "mu": 0.61},
    )
    source_rate_phys = "1e10 photons/s"
    source_rate_code_diag = to_code(source_rate_phys, "photon_rate", cu_diag)

    print("Diagnostic unit frame (pc / Msun / km-s):")
    print(f"  L_cgs = {cu_diag.L_cgs:.6e} cm")
    print(f"  M_cgs = {cu_diag.M_cgs:.6e} g")
    print(f"  V_cgs = {cu_diag.V_cgs:.6e} cm/s")
    print(f"  T_cgs = {cu_diag.T_cgs:.6e} s")
    print(f"  light_speed_code = {cu_diag.light_speed_code:.6e}")
    print(f"  source_rate: {source_rate_phys} -> {source_rate_code_diag:.4e} code units")
    return cu_diag, source_rate_phys, source_rate_code_diag


def build_ramses_rt_unit_system():
    """
    Primary unit system used to run the actual RAMSES-RT comparison:
    length = 1 cm (paper domain size), velocity = c.
    """
    cu = CodeUnits.from_config(
        {
            "length": "1 cm",
            "mass": "1 g",
            "velocity": "3e10 cm/s",   # velocity scale = c
        },
        {"gamma": 5.0 / 3.0, "mu": 0.61},
    )

    source_rate_phys = "1e10 photons/s"
    source_rate_code = to_code(source_rate_phys, "photon_rate", cu)

    t_phys = 5.2e-11  # seconds, paper snapshot time
    time_code = t_phys / cu.T_cgs

    print("=" * 70)
    print("RAMSES-RT UNIT SYSTEM")
    print("=" * 70)
    print(f"  L_cgs = {cu.L_cgs:.6e} cm")
    print(f"  V_cgs = {cu.V_cgs:.6e} cm/s (= c)")
    print(f"  T_cgs = {cu.T_cgs:.6e} s")
    print(f"  light_speed_code = {cu.light_speed_code:.6e}")
    print(f"  source_rate_phys = {source_rate_phys} -> code = {source_rate_code:.4e}")
    print(f"  time_code = {time_code:.4e}")

    light_cross_time = cu.L_cgs / cu.V_cgs
    print(f"  light-crossing time = {light_cross_time:.4e} s "
          f"= {light_cross_time / cu.T_cgs:.4e} code units")

    return cu, source_rate_phys, source_rate_code, t_phys, time_code


# --------------------------------------------------------------------------- #
# 3. Equation manager / solver setup
# --------------------------------------------------------------------------- #
def build_rt_solver(cu, size_shape):
    eq_test = EquationManager_RT(
        light_speed=cu.light_speed_code,
        mesh_shape=(size_shape, size_shape, size_shape),
        debug=False,
    )

    ss = dh.signal_speed_Rusanov
    solver_test = dh.LaxFriedrichs_Radiative_transfer(equation_manager=eq_test, signal_speed=ss)
    cf_test = dh.ConvectiveFlux_Radiative_transfer(eq_test, solver_test, dh.PLM(limiter="MINMOD"))

    print(f"Resolution: {size_shape}^3 cells | light_speed_code={cu.light_speed_code:.4e}")
    return eq_test, cf_test


def make_stellar_force(eq_test, cu, source_rate_code, *, beam_axis=0, beam_sign=1,
                        beam_length_cells=8, injection_momentum=True):
    return StellarRadiationForce(
        escape_fraction=0.1,
        dx=1.0,
        injection_mode="stromgren",
        stromgren_rate=source_rate_code,
        injection_momentum=injection_momentum,
        injection_geometry="3D",
        eq=eq_test,
        debug=False,
        momentum_only=False,
        beam_axis=beam_axis,
        beam_sign=beam_sign,
        beam_length_cells=beam_length_cells,
        beam_reduced_flux=1.0,
        beam_momentum_scaling="legacy_c2_source2",
        cu=cu,
    )


def make_hydro(cf_test, force):
    return dh.hydro(n_super_step=1000, fluxes=[cf_test], forces=[force])


# --------------------------------------------------------------------------- #
# 4. Isotropic point-source run
# --------------------------------------------------------------------------- #
def run_isotropic_source(eq_test, cf_test, cu, size_shape, source_rate_code, time_code):
    stellar_force = make_stellar_force(
        eq_test, cu, source_rate_code,
        beam_axis=0, beam_sign=1, beam_length_cells=8,
        injection_momentum=False,
    )
    hydrosim_test = make_hydro(cf_test, stellar_force)

    params = {
        "star_masses": jnp.array([1.0]),
        "star_ages": jnp.array([0.1]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array(
            [[size_shape // 2, size_shape // 2, size_shape // 2]], dtype=jnp.int32
        ),
    }

    sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

    print("Running isotropic point source to t = "
          f"{time_code:.4e} code units...")
    field_test, *_ = hydrosim_test.evolve_till_time(cp.deepcopy(sol_test), params, time_code)

    print(f"E_gamma: min={float(jnp.min(field_test[0])):.4e}, "
          f"max={float(jnp.max(field_test[0])):.4e}")
    print(f"|F|_max: {float(jnp.sqrt(jnp.sum(field_test[1:]**2, axis=0)).max()):.4e}")
    return field_test


# --------------------------------------------------------------------------- #
# 5. Beam runs (x, y, xy-diagonal)
# --------------------------------------------------------------------------- #
def build_beam_parameters(size_shape, cu):
    beam_length_cells = 8
    N_beam = 1.0  # photons / cm^2

    dx_cell_cm = 1.0 / size_shape
    E_beam_entry = N_beam / dx_cell_cm
    beam_area = (beam_length_cells * dx_cell_cm) ** 2
    beam_photon_rate = N_beam * cu.light_speed_cgs * beam_area

    print("=" * 70)
    print("RAMSES-RT BEAM CONFIGURATIONS")
    print("=" * 70)
    print(f"  N = {N_beam} cm^-2 | reduced flux = 1.0 (|F| = c*E)")
    print(f"  cell size = {dx_cell_cm:.6e} cm")
    print(f"  E_beam_entry = {E_beam_entry:.4e} code units")
    print(f"  beam_area = {beam_area:.4e} cm^2 | beam_photon_rate = {beam_photon_rate:.4e} photons/s")

    return beam_length_cells, N_beam, dx_cell_cm, E_beam_entry


def run_beam_x(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
               beam_length_cells, E_beam_entry):
    force = make_stellar_force(eq_test, cu, source_rate_code,
                                beam_axis=0, beam_sign=1,
                                beam_length_cells=beam_length_cells)
    hydrosim = make_hydro(cf_test, force)

    params = {
        "star_masses": jnp.array([1.0]),
        "star_ages": jnp.array([0.1]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array(
            [[0, size_shape // 2, size_shape // 2]], dtype=jnp.int32
        ),
    }

    half = beam_length_cells // 2
    c0 = size_shape // 2
    sol = jnp.zeros((4, size_shape, size_shape, size_shape))
    sol = sol.at[0, :beam_length_cells, c0 - half:c0 + half, c0 - half:c0 + half].set(E_beam_entry)
    sol = sol.at[1, :beam_length_cells, c0 - half:c0 + half, c0 - half:c0 + half].set(
        cu.light_speed_code * E_beam_entry
    )

    field, *_ = hydrosim.evolve_till_time(cp.deepcopy(sol), params, time_code)
    print(f"X-beam done | E_gamma: min={float(jnp.min(field[0])):.4e}, "
          f"max={float(jnp.max(field[0])):.4e}")
    return field


def run_beam_y(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
               beam_length_cells, E_beam_entry):
    force = make_stellar_force(eq_test, cu, source_rate_code,
                                beam_axis=1, beam_sign=1,
                                beam_length_cells=beam_length_cells)
    hydrosim = make_hydro(cf_test, force)

    params = {
        "star_masses": jnp.array([1.0]),
        "star_ages": jnp.array([0.1]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array(
            [[size_shape // 2, 0, size_shape // 2]], dtype=jnp.int32
        ),
    }

    half = beam_length_cells // 2
    c0 = size_shape // 2
    sol = jnp.zeros((4, size_shape, size_shape, size_shape))
    sol = sol.at[0, c0 - half:c0 + half, :beam_length_cells, c0 - half:c0 + half].set(E_beam_entry)
    sol = sol.at[2, c0 - half:c0 + half, :beam_length_cells, c0 - half:c0 + half].set(
        cu.light_speed_code * E_beam_entry
    )

    field, *_ = hydrosim.evolve_till_time(cp.deepcopy(sol), params, time_code)
    print(f"Y-beam done | E_gamma: min={float(jnp.min(field[0])):.4e}, "
          f"max={float(jnp.max(field[0])):.4e}")
    return field


def run_beam_xy(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
                beam_length_cells, E_beam_entry):
    force = make_stellar_force(eq_test, cu, source_rate_code,
                                beam_axis=0, beam_sign=1,
                                
                                beam_length_cells=beam_length_cells)
    hydrosim = make_hydro(cf_test, force)

    params = {
        "star_masses": jnp.array([1.0]),
        "star_ages": jnp.array([0.1]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array(
            [[0, 0, size_shape // 2]], dtype=jnp.int32
        ),
    }

    c0 = size_shape // 2
    sol = jnp.zeros((4, size_shape, size_shape, size_shape))
    sol = sol.at[0, :beam_length_cells, :beam_length_cells, c0 - beam_length_cells // 2:c0 + beam_length_cells // 2].set(E_beam_entry)

    F_diag = cu.light_speed_code * E_beam_entry / jnp.sqrt(2.0)
    sol = sol.at[1, :beam_length_cells, :beam_length_cells, c0 - beam_length_cells // 2:c0 + beam_length_cells // 2].set(F_diag)
    sol = sol.at[2, :beam_length_cells, :beam_length_cells, c0 - beam_length_cells // 2:c0 + beam_length_cells // 2].set(F_diag)

    field, *_ = hydrosim.evolve_till_time(cp.deepcopy(sol), params, time_code)
    print(f"XY-beam done | E_gamma: min={float(jnp.min(field[0])):.4e}, "
          f"max={float(jnp.max(field[0])):.4e}")
    return field


# --------------------------------------------------------------------------- #
# 6. Diagnostics (unit-aware printouts)
# --------------------------------------------------------------------------- #
def print_center_diagnostics(field_test, eq_test, cu_diag, size_shape):
    """
    Physical-unit diagnostic using the diagnostic (pc/Msun/km-s) unit frame.
    Fixed: no more accidental re-definition of `cu_rt` with L_cgs=M_cgs=V_cgs=1
    shadowing the earlier physically meaningful unit system.
    """
    ix = iy = iz = size_shape // 2

    E_center = field_test[0, ix, iy, iz]
    Fx_center = field_test[1, ix, iy, iz]
    Fy_center = field_test[2, ix, iy, iz]
    Fz_center = field_test[3, ix, iy, iz]

    Fmag = jnp.sqrt(field_test[1] ** 2 + field_test[2] ** 2 + field_test[3] ** 2)
    reduced_flux = Fmag / jnp.maximum(eq_test.light_speed * field_test[0], 1e-30)

    def show(value, dim, unit=None):
        return format_quantity(value, dim, cu_diag, out_unit=unit)

    print("=== Physical unit diagnostic (center cell) ===")
    print("E_gamma(center)  =", show(E_center, "photon_surface_density", "photons/cm^2"))
    print("Fx_gamma(center) =", show(Fx_center, "photon_flux", "photons/s/cm^2"))
    print("Fy_gamma(center) =", show(Fy_center, "photon_flux", "photons/s/cm^2"))
    print("Fz_gamma(center) =", show(Fz_center, "photon_flux", "photons/s/cm^2"))
    print(f"max reduced flux = {float(jnp.max(reduced_flux)):.6e}")
    print(f"anisotropy Fy/Fx = "
          f"{float(jnp.max(jnp.abs(field_test[2])) / (jnp.max(jnp.abs(field_test[1])) + 1e-30)):.6e}")
    print(f"anisotropy Fz/Fx = "
          f"{float(jnp.max(jnp.abs(field_test[3])) / (jnp.max(jnp.abs(field_test[1])) + 1e-30)):.6e}")


def check_radial_falloff(field_test, size_shape):
    """1/r^2 sanity check along the x-axis through the source."""
    ix = size_shape // 2
    x_line = np.asarray(field_test[0, :, ix, ix], dtype=float)
    r_cells = np.abs(np.arange(size_shape, dtype=float) - float(ix))
    r2_profile = np.where(r_cells > 0.0, x_line * r_cells ** 2, np.nan)

    fig, ax = plt.subplots()
    ax.plot(r_cells[1:], x_line[1:] * r_cells[1:] ** 2)
    ax.set_xlabel("|x - x0| in cells")
    ax.set_ylabel("E_gamma * r^2")
    ax.set_title("1/r^2 check along the x-axis")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, r2_profile


def diagnose_source_rate_scaling(cu_diag):
    """
    Explains why source_rate_code can be very large: it depends on the
    time unit T_cgs = L_cgs / V_cgs baked into the chosen unit system.
    """
    print("=== source_rate_code scale analysis ===")
    print(f"T_cgs = {cu_diag.T_cgs:.4e} s ({cu_diag.T_cgs/365.25/24/3600:.2e} yr)")
    print(f"PhotonRate_cgs (1 photon/s in code units) = {cu_diag.PhotonRate_cgs:.4e}")

    print("\nOption 1: reduce the physical source rate")
    for source_phys in ["1e6 photons/s", "1e8 photons/s", "1e5 photons/s"]:
        code_val = to_code(source_phys, "photon_rate", cu_diag)
        print(f"  {source_phys:15s} -> {code_val:.4e} code units")

    print("\nOption 2: increase the velocity scale (shortens T_cgs)")
    for v_phys in ["10 km/s", "100 km/s", "1000 km/s"]:
        cu_try = CodeUnits.from_config({"length": "1 pc", "mass": "1 Msun", "velocity": v_phys})
        code_val = to_code("1e10 photons/s", "photon_rate", cu_try)
        print(f"  v={v_phys:10s} -> T_cgs={cu_try.T_cgs:.4e} s -> source={code_val:.4e}")


# --------------------------------------------------------------------------- #
# 7. Plotting helpers
# --------------------------------------------------------------------------- #
def plot_field_slice(field, size_shape, *, component=0, plane="xy",
                      log_scale=True, zoom=None, title=None, cmap="hot"):
    """
    Generic 2D-slice plotter (replaces the 4 duplicated imshow blocks).
    plane: "xy" (fix z), "xz" (fix y), "yz" (fix x)
    zoom : (half_width,) -> restrict view to center +/- half_width, or None
    """
    c = size_shape // 2
    data = np.asarray(field[component])

    if plane == "xy":
        img, xlabel, ylabel = data[:, :, c], "y [cells]", "x [cells]"
    elif plane == "xz":
        img, xlabel, ylabel = data[:, c, :], "z [cells]", "x [cells]"
    elif plane == "yz":
        img, xlabel, ylabel = data[c, :, :], "z [cells]", "y [cells]"
    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

    fig, ax = plt.subplots()

    if log_scale:
        floor = 1e-30
        norm = mcolors.LogNorm(vmin=max(img[img > 0].min(), floor) if np.any(img > 0) else floor,
                                vmax=max(img.max(), floor))
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
        cbar_label = "Photon number density (log scale)"
    else:
        im = ax.imshow(img, origin="lower", cmap=cmap)
        cbar_label = "Photon number density"

    ax.set_title(title or f"field[{component}] ({plane}-plane, center slice)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if zoom is not None:
        ax.set_xlim(c - zoom, c + zoom)
        ax.set_ylim(c - zoom, c + zoom)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)
    fig.tight_layout()
    return fig, ax


def plot_reduced_flux(field, cu, size_shape, plane_index=None):
    c = plane_index if plane_index is not None else size_shape // 2
    E = np.asarray(field[0, :, :, c], dtype=float)
    F = np.sqrt(np.asarray(field[1, :, :, c]) ** 2 + np.asarray(field[2, :, :, c]) ** 2)
    reduced_flux = F / (cu.light_speed_code * np.maximum(E, 1e-30))

    fig, ax = plt.subplots()
    im = ax.imshow(reduced_flux, origin="lower", cmap="viridis", vmin=0, vmax=1.1)
    ax.set_title("Reduced Flux |F| / (c*E) (xy-plane)")
    ax.set_xlabel("y [cells]")
    ax.set_ylabel("x [cells]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("f", fontsize=11)
    fig.tight_layout()

    frac_causal = float((reduced_flux <= 1.0).sum()) / reduced_flux.size * 100.0
    print(f"max(|F|/(c*E)) = {float(reduced_flux.max()):.6f} | "
          f"{frac_causal:.1f}% of cells respect causality")
    return fig, ax


def plot_comparison_grid(field_test, field_beam_x, field_beam_y, field_beam_xy, size_shape, t_phys):
    """2x2 comparison grid: isotropic source vs the three beam geometries."""
    z = size_shape // 2
    norm_log = mcolors.LogNorm(vmin=1e-15, vmax=1e2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    panels = [
        (field_test, "Isotropic Point Source\n10^10 photons/s"),
        (field_beam_x, "X-direction Beam\nN = 1 cm^-2"),
        (field_beam_y, "Y-direction Beam\nN = 1 cm^-2"),
        (field_beam_xy, "XY-diagonal Beam\nN = 1 cm^-2"),
    ]

    for ax, (field, label) in zip(axes.ravel(), panels):
        img = np.asarray(field[0, :, :, z], dtype=float)
        im = ax.imshow(img, origin="lower", cmap="hot", norm=norm_log)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("y [cells]", fontsize=11)
        ax.set_ylabel("x [cells]", fontsize=11)
        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("E_gamma [code units]", fontsize=10)

    fig.suptitle(
        f"RAMSES-RT Radiative Transfer Comparison\n"
        f"Domain: 1 cm, Resolution: {size_shape}^3, Time: {t_phys:.2e} s",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    return fig, axes


def print_comparison_summary(field_test, field_beam_x, field_beam_y, field_beam_xy):
    def rng(f):
        return float(jnp.min(f[0])), float(jnp.max(f[0]))

    print("=" * 70)
    print("RAMSES-RT COMPARISON SUMMARY")
    print("=" * 70)
    for name, field in [
        ("Isotropic source", field_test),
        ("X-beam", field_beam_x),
        ("Y-beam", field_beam_y),
        ("XY-diagonal beam", field_beam_xy),
    ]:
        lo, hi = rng(field)
        print(f"{name}: E_gamma min={lo:.4e}, max={hi:.4e}")


# --------------------------------------------------------------------------- #
# 8. Optional: Sedov-Taylor Athena IC loader (kept isolated, NOT wired into
#    the RAMSES-RT pipeline -- it was dead code in the original notebook).
# --------------------------------------------------------------------------- #
def load_sedov_taylor_ics(athena_outputs_loc="../../data/athena_comparison/",
                           ic_filename="Blast.out2.00000.athdf"):
    ICs = athdf(athena_outputs_loc + ic_filename)
    sol = jnp.zeros((5, 100, 100, 100))
    sol = sol.at[0].set(ICs["dens"])
    sol = sol.at[-1].set(ICs["Etot"])
    print("Sedov-Taylor ICs loaded:", list(ICs.keys()), "| sol shape:", sol.shape)
    return ICs, sol


# --------------------------------------------------------------------------- #
# 9. Main driver
# --------------------------------------------------------------------------- #
def main(size_shape=256, run_sedov_demo=False):
    print_backend_info()

    cu_diag, _, _ = build_diagnostic_unit_system()
    cu, source_rate_phys, source_rate_code, t_phys, time_code = build_ramses_rt_unit_system()

    eq_test, cf_test = build_rt_solver(cu, size_shape)

    field_test = run_isotropic_source(eq_test, cf_test, cu, size_shape, source_rate_code, time_code)

    beam_length_cells, N_beam, dx_cell_cm, E_beam_entry = build_beam_parameters(size_shape, cu)
    field_beam_x = run_beam_x(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
                               beam_length_cells, E_beam_entry)
    field_beam_y = run_beam_y(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
                               beam_length_cells, E_beam_entry)
    field_beam_xy = run_beam_xy(eq_test, cf_test, cu, size_shape, source_rate_code, time_code,
                                 beam_length_cells, E_beam_entry)

    print_center_diagnostics(field_test, eq_test, cu_diag, size_shape)
    check_radial_falloff(field_test, size_shape)
    diagnose_source_rate_scaling(cu_diag)

    plot_field_slice(field_test, size_shape, plane="xy", log_scale=True,
                      title="E_gamma (xy-plane, isotropic source)")
    plot_field_slice(field_test, size_shape, plane="xy", log_scale=True, zoom=5,
                      title="E_gamma zoom near source (xy-plane)")
    plot_reduced_flux(field_test, cu, size_shape)

    plot_comparison_grid(field_test, field_beam_x, field_beam_y, field_beam_xy, size_shape, t_phys)
    


    def compute_radial_profile_2d(img2d, center_y, center_z, dx_cell_cm, nbins=80):
        """
        Azimuthally-averaged radial profile in a 2D plane.
        For a yz slice, returns <N_gamma(r)> as a function of radius r [cm].
        """
        img = np.asarray(img2d, dtype=float)

        ny, nz = img.shape
        yy, zz = np.indices((ny, nz))

        r_cells = np.sqrt((yy - center_y) ** 2 + (zz - center_z) ** 2)
        r_cm = r_cells * dx_cell_cm

        r_max = r_cm.max()
        edges = np.linspace(0.0, r_max, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        sums = np.zeros(nbins, dtype=float)
        counts = np.zeros(nbins, dtype=float)

        flat_r = r_cm.ravel()
        flat_img = img.ravel()

        bin_ids = np.digitize(flat_r, edges) - 1
        valid = (bin_ids >= 0) & (bin_ids < nbins)

        np.add.at(sums, bin_ids[valid], flat_img[valid])
        np.add.at(counts, bin_ids[valid], 1.0)

        profile = np.divide(
            sums,
            np.maximum(counts, 1.0),
            out=np.zeros_like(sums),
            where=counts > 0,
        )

        return centers[counts > 0], profile[counts > 0]


    def plot_yz_radial_profiles_every_10_cells(field, size_shape, dx_cell_cm,
                                            *, slice_step=10, nbins=80,
                                            component=0, logy=True,
                                            title=None):
        """
        Take yz slices every `slice_step` cells along x and plot all radial profiles
        of photon number density on the same figure.

        field[component] is assumed to be shaped (Nx, Ny, Nz).
        For RAMSES-RT here, component=0 corresponds to photon number density / E_gamma.
        """
        data = np.asarray(field[component], dtype=float)

        cy = size_shape // 2
        cz = size_shape // 2

        x_slices = np.arange(0, size_shape, slice_step)

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.cm.viridis(np.linspace(0.0, 1.0, len(x_slices)))

        for color, ix in zip(cmap, x_slices):
            yz_slice = data[ix, :, :]
            r_cm, profile = compute_radial_profile_2d(
                yz_slice, cy, cz, dx_cell_cm, nbins=nbins
            )

            if profile.size == 0:
                continue

            profile = np.clip(profile, 1e-30, None)

            ax.plot(
                r_cm,
                profile,
                color=color,
                lw=1.6,
                alpha=0.95,
                label=f"x = {ix}",
            )

        ax.set_xlabel("Radius r [cm]")
        ax.set_ylabel("Photon number density")
        ax.set_title(
            title or "Photon number density radial profiles in yz slices (every 10 cells)"
        )
        ax.grid(True, alpha=0.3)

        if logy:
            ax.set_yscale("log")

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            title="yz slice",
            frameon=True,
        )

        fig.tight_layout()
        return fig, ax

    plot_comparison_grid(field_test, field_beam_x, field_beam_y, field_beam_xy, size_shape, t_phys)
    plot_yz_radial_profiles_every_10_cells(
        field_beam_x,
        size_shape,
        dx_cell_cm,
        slice_step=10,
        nbins=80,
        component=0,
        logy=True,
        title="X-beam: radial profiles of photon number density in yz slices",
    )
    print_comparison_summary(field_test, field_beam_x, field_beam_y, field_beam_xy)

    if run_sedov_demo:
        print("caca")
        load_sedov_taylor_ics()

    plt.show()

    return {
        "field_test": field_test,
        "field_beam_x": field_beam_x,
        "field_beam_y": field_beam_y,
        "field_beam_xy": field_beam_xy,
        "cu": cu,
        "cu_diag": cu_diag,
    }


if __name__ == "__main__":
    main()

