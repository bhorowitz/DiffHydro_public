
# Depending on your system/cluster, you may want to specify which GPU you want to use
import os, sys, importlib, atexit
sys.path.append("../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# RUN PARAMETERS (defined first, needed to build the output folder name)
# ============================================================================
velocity_ref = 1e10
velocity_new = 10000.0   # <-- change this value
mass_unit_ref = 1.0
mass_unit_new = 1.0
time_phys_global = 1.3e-11

C_CGS = 2.99792458e10  # cm/s, exact speed of light


# ============================================================================
# OUTPUT FOLDER: named with arrival time + velocity parameters
# ============================================================================
BASE_OUTPUT_DIR = "examples/athena/Images_athena"

def make_run_folder_name(time_phys, velocity_cms):
    return f"t{time_phys:.1e}s_v{velocity_cms:.1e}cms"

master_folder_name = (
    f"compare_t{time_phys_global:.1e}s_"
    f"vref{velocity_ref:.1e}cms_vnew{velocity_new:.1e}cms"
)
master_output_dir = os.path.join(BASE_OUTPUT_DIR, master_folder_name)
os.makedirs(master_output_dir, exist_ok=True)

run_ref_output_dir = os.path.join(master_output_dir, make_run_folder_name(time_phys_global, velocity_ref))
run_new_output_dir = os.path.join(master_output_dir, make_run_folder_name(time_phys_global, velocity_new))
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

    mask = (r_sph > injection_radius) & (r_sph < 23) #(r_sph < radius_truncation)
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


def run_pipeline(velocity_cms, mass_unit_g, tag, output_dir, time_phys=5.2e-11):
    print("\\n" + "=" * 80)
    print(f"RUN PIPELINE: {tag} | velocity unit = {velocity_cms:.6e} cm/s")
    print(f"Output dir  = {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================================
    # GRID GEOMETRY (single source of truth, no duplication)
    # ============================================================================
    size_shape       = int(os.environ.get("N", 256))
    box_width_phys   = 1.0                          # cm : physical width of the whole box
    dx_phys          = box_width_phys / size_shape   # cm per cell
    cell_size_phys   = dx_phys                       # alias kept for downstream code
    source_rate_phys = 1e10                          # photons / s

    # ct doit rester dans la boite (periodique) : ct < box/2 -> t < box/(2c).
    t_phys = time_phys

    # --- code units: length unit = ONE CELL, velocity unit = velocity_cms ------
    cu = CodeUnits.from_config(
        {
            "length": f"{dx_phys} cm",
            "mass": f"{mass_unit_g} g",
            "velocity": f"{velocity_cms} cm/s",
        },
        {"gamma": 5.0 / 3.0, "mu": 0.61},
    )

    light_speed_code = C_CGS / cu.V_cgs      # vitesse de la lumiere en unites code
    dx_cell_code     = 1.0                   # le solveur suppose dx = 1 cellule = 1 unite code
    time_code        = t_phys / cu.T_cgs
    source_rate_code = source_rate_phys * cu.T_cgs

    print("Snapshot time:", f"{t_phys:.2e} s")

    print("\\nCode unit scales:")
    print(f"  L_cgs = {cu.L_cgs:.6e} cm")
    print(f"  V_cgs = {cu.V_cgs:.6e} cm/s")
    print(f"  T_cgs = {cu.T_cgs:.6e} s")
    print(f"  light_speed_code = {light_speed_code:.6e}")

    print("\\nGrid geometry:")
    print(f"  box_width_phys = {box_width_phys:.6e} cm")
    print(f"  size_shape = {size_shape}")
    print(f"  cell_size_phys (= dx_phys) = {cell_size_phys:.6e} cm")
    print(f"  dx_cell_code = {dx_cell_code:.6e}")

    print("\\nCode values:")
    print(f"  source_rate_phys = {source_rate_phys:.4e} photons/s")
    print(f"  source_rate_code = {source_rate_code:.4e}")
    print(f"  time_code = {time_code:.4e}")

    light_cross_time_phys = box_width_phys / C_CGS
    print(f"\\nPhysical light-crossing time of box = {light_cross_time_phys:.4e} s")
    if t_phys >= box_width_phys / (2.0 * C_CGS):
        print(f"  WARNING: ct >= box/2, le front lumineux boucle probablement autour du domaine periodique.")

    # ============================================================================
    # SIMULATION SETUP
    # ============================================================================
    eq_test = EquationManager_RT(
        light_speed=light_speed_code,
        mesh_shape=(size_shape, size_shape, size_shape),
        debug=False,
    )

    ss = dh.signal_speed_Rusanov
    solver_test = dh.LaxFriedrichs_Radiative_transfer(
        equation_manager=eq_test,
        signal_speed=ss
    )

    cf_test = dh.ConvectiveFlux_Radiative_transfer(
        eq_test,
        solver_test,
        dh.PLM(limiter="VANLEER"),
    )

    stellar_force = StellarRadiationForce(
        escape_fraction=0.1,
        dx=dx_cell_code,               # = 1: uniquement utilise comme volume de cellule
        injection_mode="stromgren",
        stromgren_rate=source_rate_code,
        injection_momentum=False,
        gaussian_star=True,
        injection_geometry="3D",
        eq=eq_test,
        debug=False,
        momentum_only=False,
        beam_axis=0,
        beam_sign=1,
        beam_length_cells=0,
        beam_reduced_flux=0,
        beam_momentum_scaling="physical",
    )

    hydrosim_test = dh.hydro(
        n_super_step=20000,
        fluxes=[cf_test],
        forces=[stellar_force],
    )

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

    field_test, _, _, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(
        cp.deepcopy(sol_test),
        params,
        time_code,
    )

    print("Simulation completed!")
    print(f"  Number of steps: {nombre_de_pas_test_test}")
    print(f"  Time step history physical: {float(jnp.sum(dt_historique_test)) * cu.T_cgs:.4e} s")

    # ============================================================================
    # DIAGNOSTICS
    # ============================================================================
    dt_total_phys = float(jnp.sum(dt_historique_test)) * cu.T_cgs
    N_expected = source_rate_phys * dt_total_phys
    N_gamma_tot = float(np.sum(np.asarray(field_test[0])))

    print(f"Expected injected photons = {N_expected:.4e} photons")
    print(f"Total photons in domain   = {N_gamma_tot:.4e} photons")
    print(f"Difference                = {N_gamma_tot - N_expected:.4e} photons")

    extent_phys = compute_extent_phys(size_shape, cell_size_phys)
    center_phys = box_width_phys / 2
    zoom_half_size = 0.05
    xlim_zoom = (center_phys - zoom_half_size, center_phys + zoom_half_size)
    ylim_zoom = (center_phys - zoom_half_size, center_phys + zoom_half_size)

    E3d = np.asarray(field_test[0])
    E_slice = E3d[:, :, size_shape // 2]
    E_slice_density = E_slice
    N_col_phys = np.sum(E3d, axis=2)

    # ============================================================================
    # FINAL PLOTS FOR THIS RUN
    # ============================================================================
    make_black_imshow(
        data=E_slice_density,
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
        vmin=-20,
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
        "velocity_cms": velocity_cms,
        "cu": cu,
        "field_3d": E3d,
        "E_slice": E_slice_density,
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
    velocity_ref, mass_unit_ref,
    f"run_ref_{time_phys_global:.1e}s", run_ref_output_dir, time_phys_global
)
run_new = run_pipeline(
    velocity_new, mass_unit_new,
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

# Reutilise le systeme non centre calcule dans run_pipeline pour run_ref
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
    title=f"FINAL NEW VELOCITY - Photon number density near source ({time_phys_global:.1e}s)",
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
    cbar_label=r"$\\Delta n_\\gamma$",
    cmap="coolwarm",
    log=False,
    output_dir=master_output_dir,
)

make_black_imshow(
    data=diff_rel,
    extent=extent_phys,
    title=f"RELATIVE DIFFERENCE - (new-ref)/ref ({time_phys_global:.1e}s)",
    cbar_label=r"$\\Delta n_\\gamma / n_{\\gamma,\\mathrm{ref}}$",
    cmap="coolwarm",
    log=False,
    output_dir=master_output_dir,
)
