
# Depending on your system/cluster, you may want to specify which GPU you want to use
import os, sys, importlib, atexit
sys.path.append("../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

import jax
jax.config.update("jax_disable_jit", False)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# core diffhydro
import diffhydro as dh

os.environ["DIFFHYDRO_DEBUG_CHECKS"] = "False"
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

from diffhydro.units import CodeUnits, to_code, from_code_fields, to_code_fields
from diffhydro.equationmanager_radiative_transf_no_chat import EquationManager as EquationManager_RT
from diffhydro.physics.radiative_transfer import StellarRadiationForce
import copy as cp

# ============================================================================
# PLOT STYLE: BLACK BACKGROUND
# ============================================================================
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "axes.titlecolor": "white",
    "grid.color": "gray",
})


# ============================================================================
# RUN PARAMETERS
# ============================================================================
# --- Physical (fixed) code-unit anchors -- SAME for every run -------------
# KEY POINT (from the reference script):
# The solver is hard-wired to dx = 1 cell (hydro.dx_o = 1.0, cf.dx_o = 1).
# So the code LENGTH unit must be ONE CELL (box_width / N), not the box.
# The code VELOCITY unit must be the true speed of light, so that
# light_speed_code = c_cgs / cu.V_cgs == 1 exactly, and the CFL condition
# (dt = cfl / (ndim * c / dx_o)) is meaningful and comparable across runs.
C_CGS = 2.99792458e10  # cm/s, exact speed of light

size_shape        = int(os.environ.get("N", 256))
box_width_phys    = 1.0                              # cm
dx_phys           = box_width_phys / size_shape       # cm per cell -- code length unit
mass_unit_g       = 1.0                                # g -- code mass unit (same for all runs)
source_rate_phys  = 1e10                               # photons / s
time_phys_global  = 1.3e-11                             # s (snapshot time, same for all runs)

# --- What actually differs between the two runs ----------------------------
# NOTE: in the very first version of this pipeline, "velocity_ref"/"velocity_new"
# were (incorrectly) fed into CodeUnits as the VELOCITY UNIT itself, which
# silently changed T_cgs/V_cgs between runs and made the comparison meaningless
# (see chat history). Since the code-unit velocity must always be c_cgs (see
# KEY POINT above), these two values are now used as the *physical light speed
# actually used inside the radiative solver* (a "reduced speed of light",
# a.k.a. RSLA -- Reduced Speed of Light Approximation), which is a real and
# common numerical trick in RT codes to speed up convergence. This keeps the
# units homogeneous while still giving two genuinely different, comparable runs.
light_speed_used_ref_cms = 1e10     # cm/s -- "reference" signal speed used in the solver
light_speed_used_new_cms = 10000.0  # cm/s -- "reduced" signal speed used in the solver


# ============================================================================
# OUTPUT FOLDER: named with arrival time + light-speed parameters
# ============================================================================
BASE_OUTPUT_DIR = "examples/athena/Images_athena"

def make_run_folder_name(time_phys, light_speed_cms):
    return f"t{time_phys:.1e}s_c{light_speed_cms:.1e}cms"

master_folder_name = (
    f"compare_t{time_phys_global:.1e}s_"
    f"cref{light_speed_used_ref_cms:.1e}cms_cnew{light_speed_used_new_cms:.1e}cms"
)
master_output_dir = os.path.join(BASE_OUTPUT_DIR, master_folder_name)
os.makedirs(master_output_dir, exist_ok=True)

run_ref_output_dir = os.path.join(master_output_dir, make_run_folder_name(time_phys_global, light_speed_used_ref_cms))
run_new_output_dir = os.path.join(master_output_dir, make_run_folder_name(time_phys_global, light_speed_used_new_cms))
os.makedirs(run_ref_output_dir, exist_ok=True)
os.makedirs(run_new_output_dir, exist_ok=True)


# ============================================================================
# LOGGING: one log file per full script launch, stored in the master folder
# ============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


log_path = os.path.join(master_output_dir, "run_log.txt")
log_file = open(log_path, "w")  # "w" ecrase le fichier a chaque lancement

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

atexit.register(log_file.close)

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
print(f"Master output folder: {master_output_dir}")
print(f"Run ref output folder: {run_ref_output_dir}")
print(f"Run new output folder: {run_new_output_dir}")


def compute_extent_phys(size_shape, cell_size_phys):
    """
    Convertit les indices de pixels [0, size_shape] en unites physiques,
    pixel 0 -> 0, dernier pixel -> size_shape * cell_size_phys.
    Aucun centrage sur zero.
    """
    box_extent = size_shape * cell_size_phys
    return [0, box_extent, 0, box_extent]


def make_black_imshow(data, extent=None, title="", cbar_label="", cmap="hot", log=False,
                      xlim=None, ylim=None, figsize=(6, 5),
                      output_dir=BASE_OUTPUT_DIR,
                      use_extent=True,
                      xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")

    imshow_kwargs = dict(
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )

    if use_extent and extent is not None:
        imshow_kwargs["extent"] = extent
        default_xlabel, default_ylabel = "y [cm]", "x [cm]"
    else:
        default_xlabel, default_ylabel = "y [cell]", "x [cell]"

    if log:
        data_masked = np.ma.masked_less_equal(data, 0.0)
        positive = np.asarray(data)[np.asarray(data) > 0]
        vmin = positive.min() if positive.size else 1e-200
        vmax = positive.max() if positive.size else 1.0
        im = ax.imshow(
            data_masked,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            **imshow_kwargs,
        )
    else:
        im = ax.imshow(
            data,
            **imshow_kwargs,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(f"{output_dir}/{safe_title}.png", dpi=300, bbox_inches="tight")
    # plt.show()


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

    mask = (r_sph > injection_radius) & (r_sph < radius_truncation)
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

    c, b = float(popt[0]), float(popt[1])
    b_err = float(np.sqrt(pcov[1, 1])) if pcov.size else np.nan

    y_pred = np.exp(line_model(log_r, c, b))

    print(f"[{tag}] c={c:.6f}  b={b:.6f} (+/- {b_err:.2e})")
    print(f"[{tag}] injection_radius={injection_radius}, fit_range=[{r_valid.min():.1f}, {r_valid.max():.1f}] cells")

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
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loglog_spherical_average_{tag}.png", dpi=300, bbox_inches="tight")
    # plt.show()

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
    # plt.show()

    return {
        "tag": tag,
        "r_sph": r_sph,
        "y_sph": y_sph,
        "r_valid": r_valid,
        "y_valid": y_valid,
        "x_valid": x_valid,
        "y_pred": y_pred,
        "c": c,
        "b": b,
        "b_err": b_err,
        "injection_radius": injection_radius,
        "cell_size_phys": cell_size_phys,
    }


def run_pipeline(light_speed_used_cms, tag, output_dir, time_phys=5.2e-11):
    print("\\n" + "=" * 80)
    print(f"RUN PIPELINE: {tag} | light speed used in solver = {light_speed_used_cms:.6e} cm/s")
    print(f"Output dir  = {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # CODE UNITS -- transplanted from the reference script, SAME for every run
    # ------------------------------------------------------------------------
    # length unit = ONE CELL (dx_phys), velocity unit = TRUE speed of light.
    # This makes light_speed_code == 1.0 exactly, independently of the run,
    # and guarantees dt_code, source_rate_code, time_code are all on a
    # consistent, physically meaningful scale across both runs.
    # ========================================================================
    t_phys = time_phys  # s

    cu = CodeUnits.from_config(
        {"length": f"{dx_phys} cm", "mass": f"{mass_unit_g} g", "velocity": f"{C_CGS} cm/s"},
        {"gamma": 5.0 / 3.0, "mu": 0.61},
    )

    light_speed_code_true = C_CGS / cu.V_cgs           # should be ~1.0 (sanity check)
    dx_cell_code          = 1.0                        # the solver assumes this
    time_code             = t_phys / cu.T_cgs
    source_rate_code      = source_rate_phys * cu.T_cgs

    # The value actually fed to the RT solver for THIS run (RSLA-style test):
    light_speed_code_used = light_speed_used_cms / cu.V_cgs

    print("Snapshot time:", f"{t_phys:.2e} s")

    print("\\nCode unit scales (identical across runs):")
    print(f"  L_cgs = {cu.L_cgs:.6e} cm  (= 1 cell)")
    print(f"  V_cgs = {cu.V_cgs:.6e} cm/s (= true speed of light)")
    print(f"  T_cgs = {cu.T_cgs:.6e} s")
    print(f"  light_speed_code_true  = {light_speed_code_true:.6f}  (sanity check, should be ~1)")
    print(f"  light_speed_code_used  = {light_speed_code_used:.6e}  (this run's solver light speed)")

    print("\\nGrid geometry:")
    print(f"  box_width_phys = {box_width_phys:.6e} cm")
    print(f"  size_shape = {size_shape}")
    print(f"  cell_size_phys (= dx_phys) = {dx_phys:.6e} cm")
    print(f"  dx_cell_code = {dx_cell_code:.6e}")

    print("\\nCode values:")
    print(f"  source_rate_phys = {source_rate_phys:.4e} photons/s")
    print(f"  source_rate_code = {source_rate_code:.4e}")
    print(f"  time_code = {time_code:.4e}")

    ct_expected_phys = light_speed_used_cms * t_phys
    light_cross_time_phys = box_width_phys / light_speed_used_cms
    print(f"\\nExpected front  c_used*t = {ct_expected_phys:.4e} cm "
          f"= {ct_expected_phys / dx_phys:.1f} cells "
          f"= {ct_expected_phys / box_width_phys:.3f} box widths")
    print(f"Physical light-crossing time of box (at c_used) = {light_cross_time_phys:.4e} s")
    if t_phys >= box_width_phys / (2.0 * light_speed_used_cms):
        print("  WARNING: c_used*t >= box/2, le front boucle probablement autour du domaine periodique.")

    cell_size_phys = dx_phys  # alias kept for downstream code

    # ========================================================================
    # SIMULATION SETUP -- solver/flux config transplanted from reference script
    # ========================================================================
    eq_test = EquationManager_RT(
        light_speed=light_speed_code_used,
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
    print("expected dt_code   =", eq_test.cfl / (3.0 * light_speed_code_used / 1.0))

    params = {
        "star_masses": jnp.array([1.0]),
        "star_ages": jnp.array([0.1]),
        "star_metallicities": jnp.array([0.02]),
        "star_positions": jnp.array(
            [[size_shape // 2, size_shape // 2, size_shape // 2]],
            dtype=jnp.int32,
        ),
    }

    # field[0] = photons par cellule
    sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

    print(f"\\nRunning to t = {t_phys:.3e} s = {time_code:.2f} code units ...")
    field_test, _, _, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(
        cp.deepcopy(sol_test),
        params,
        time_code,
    )

    dt_hist = np.asarray(dt_historique_test)
    dt_sum_code = float(dt_hist[dt_hist > 0].sum())
    print("Simulation completed!")
    print(f"  Number of steps: {nombre_de_pas_test_test},  dt_code[0] = {dt_hist[0]:.6e}")
    print(f"  sum(dt) = {dt_sum_code:.4f} code = {dt_sum_code * cu.T_cgs:.4e} s")

    # ========================================================================
    # DIAGNOSTICS -- transplanted checks (photon budget + front radius)
    # ========================================================================
    dt_total_phys = dt_sum_code * cu.T_cgs
    N_expected = source_rate_phys * dt_total_phys
    E3d = np.asarray(field_test[0])
    N_gamma_tot = float(E3d.sum())

    print(f"  E min/max = {E3d.min():.4e} / {E3d.max():.4e}")
    print(f"Expected injected photons = {N_expected:.4e} photons")
    print(f"Total photons in domain   = {N_gamma_tot:.4e} photons")
    print(f"Difference                = {N_gamma_tot - N_expected:.4e} photons")

    center_idx = size_shape // 2
    line = E3d[center_idx:, center_idx, center_idx]
    peak = E3d.max()
    for th in [1e-3, 1e-6, 1e-10, 1e-15]:
        idx = np.where(line > peak * th)[0]
        r = idx.max() if idx.size else 0
        print(f"  thr {th:.0e} of peak -> radius {r:3d} cells = {r * dx_phys:.4e} cm")
    print(f"  expected free-streaming radius = {ct_expected_phys / dx_phys:.1f} cells")

    extent_phys = compute_extent_phys(size_shape, cell_size_phys)
    center_phys = box_width_phys / 2
    zoom_half_size = 0.05
    xlim_zoom = (center_phys - zoom_half_size, center_phys + zoom_half_size)
    ylim_zoom = (center_phys - zoom_half_size, center_phys + zoom_half_size)

    E_slice = E3d[:, :, size_shape // 2]
    N_col_phys = np.sum(E3d, axis=2)

    # ========================================================================
    # FINAL PLOTS FOR THIS RUN
    # ========================================================================
    make_black_imshow(
        data=E_slice,
        extent=extent_phys,
        title=fr"{tag} - Photon number density near source $n_\gamma$",
        cbar_label=r"$n_\gamma$ [arb.]",
        cmap="hot",
        log=True,
        output_dir=output_dir,
    )

    make_black_imshow(
        data=N_col_phys,
        extent=extent_phys,
        title=fr"{tag} - Projected photon surface density near source $N$",
        cbar_label=r"$N$ [arb.]",
        cmap="hot",
        log=True,
        output_dir=output_dir,
    )

    make_black_imshow(
        data=N_col_phys,
        extent=None,
        use_extent=False,
        title=fr"{tag} - Photon density near source $N$ (brut)",
        cbar_label=r"$N$ [arb.]",
        cmap="hot",
        log=True,
        output_dir=output_dir,
    )

    fit_result = analyze_inverse_r2(
        field_3d=E3d,
        size_shape=size_shape,
        cell_size_phys=cell_size_phys,
        tag=tag,
        radius_truncation=max(12, size_shape // 8),
        output_dir=output_dir,
    )

    return {
        "tag": tag,
        "light_speed_used_cms": light_speed_used_cms,
        "cu": cu,
        "field_3d": E3d,
        "E_slice": E_slice,
        "N_col_phys": N_col_phys,
        "size_shape": size_shape,
        "box_width_phys": box_width_phys,
        "cell_size_phys": cell_size_phys,
        "extent_phys": extent_phys,
        "center_phys": center_phys,
        "zoom_half_size": zoom_half_size,
        "xlim_zoom": xlim_zoom,
        "ylim_zoom": ylim_zoom,
        "fit_result": fit_result,
        "output_dir": output_dir,
    }


# ============================================================================
# RUN 1 AND RUN 2
# ============================================================================
run_ref = run_pipeline(
    light_speed_used_ref_cms,
    f"run_ref_{time_phys_global:.1e}s", run_ref_output_dir, time_phys_global
)
run_new = run_pipeline(
    light_speed_used_new_cms,
    f"run_new_{time_phys_global:.1e}s", run_new_output_dir, time_phys_global
)

# ============================================================================
# OVERLAID 1/r^2 COMPARISON (saved in the master folder, shared by both runs)
# ============================================================================
fit_ref = run_ref["fit_result"]
fit_new = run_new["fit_result"]

if fit_ref is not None and fit_new is not None:
    fig, ax = plt.subplots(figsize=(15, 5), facecolor="black")
    ax.set_facecolor("black")

    ax.plot(np.log(fit_ref["r_valid"]), np.log(fit_ref["y_valid"]), "o", color="white", ms=3, label="Ref spherical avg")
    ax.plot(np.log(fit_ref["r_valid"]), np.log(fit_ref["y_pred"]), "r-", lw=2, label=f"Ref fit: b={fit_ref['b']:.3f}")
    ax.axvline(np.log(fit_ref["injection_radius"]), color="cyan", ls="--", lw=1, label=f"Ref injection = {fit_ref['injection_radius']}")

    ax.plot(np.log(fit_new["r_valid"]), np.log(fit_new["y_valid"]), "o", color="orange", ms=3, label="New spherical avg")
    ax.plot(np.log(fit_new["r_valid"]), np.log(fit_new["y_pred"]), "-", color="purple", lw=2, label=f"New fit: b={fit_new['b']:.3f}")
    ax.axvline(np.log(fit_new["injection_radius"]), color="green", ls="--", lw=1, label=f"New injection = {fit_new['injection_radius']}")

    ax.set_xlabel("log(r [cell])")
    ax.set_ylabel("log(Shell-averaged field)")
    ax.set_title("log-log fit - spherical average: both runs")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{master_output_dir}/loglog_spherical_average_both_runs_{time_phys_global:.1e}s.png", dpi=300, bbox_inches="tight")
    # plt.show()

# ============================================================================
# DIFFERENCE PLOTS (saved in the master folder, shared by both runs)
# ============================================================================
E_ref = run_ref["E_slice"]
E_new = run_new["E_slice"]

extent_phys = run_ref["extent_phys"]
xlim_zoom = run_ref["xlim_zoom"]
ylim_zoom = run_ref["ylim_zoom"]

diff_abs = E_new - E_ref
diff_rel = np.abs(diff_abs) / np.maximum(np.abs(E_ref), 1e-30)

make_black_imshow(
    data=E_ref,
    extent=extent_phys,
    title=f"FINAL REF - Photon number density near source ({time_phys_global:.1e}s)",
    cbar_label=r"$n_\gamma$ [arb.]",
    cmap="hot",
    log=True,
    xlim=xlim_zoom,
    ylim=ylim_zoom,
    output_dir=master_output_dir,
)

make_black_imshow(
    data=E_new,
    extent=extent_phys,
    title=f"FINAL NEW LIGHT SPEED - Photon number density near source ({time_phys_global:.1e}s)",
    cbar_label=r"$n_\gamma$ [arb.]",
    cmap="hot",
    log=True,
    xlim=xlim_zoom,
    ylim=ylim_zoom,
    output_dir=master_output_dir,
)

make_black_imshow(
    data=diff_abs,
    extent=extent_phys,
    title=f"ABSOLUTE DIFFERENCE - new minus ref ({time_phys_global:.1e}s)",
    cbar_label=r"$\Delta n_\gamma$",
    cmap="coolwarm",
    log=False,
    output_dir=master_output_dir,
)

make_black_imshow(
    data=diff_rel,
    extent=extent_phys,
    title=f"RELATIVE DIFFERENCE - (new-ref)/ref ({time_phys_global:.1e}s)",
    cbar_label=r"$\Delta n_\gamma / n_{\gamma,\mathrm{ref}}$",
    cmap="coolwarm",
    log=False,
    output_dir=master_output_dir,
)
