# Depending on your system/cluster, you may want to specify which GPU you want to use
import os, sys, importlib
sys.path.append("../../")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
jax.config.update("jax_disable_jit", False)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# core diffhydro
import diffhydro as dh
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

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

from diffhydro.units import CodeUnits, to_code ,from_code_fields, to_code_fields
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

def make_black_imshow(data, extent, title, cbar_label, cmap="afmhot", log=False,
                      xlim=None, ylim=None, figsize=(6, 5),
                      output_dir="examples/athena/Images_athena"):
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")

    if log:
        data_masked = np.ma.masked_less_equal(data, 0.0)
        positive = np.asarray(data)[np.asarray(data) > 0]
        vmin = positive.min() if positive.size else 1e-30
        vmax = positive.max() if positive.size else 1.0
        im = ax.imshow(
            data_masked,
            origin="lower",
            cmap=cmap,
            extent=extent,
            aspect="equal",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            extent=extent,
            aspect="equal",
        )

    ax.set_title(title)
    ax.set_xlabel("y [cm]")
    ax.set_ylabel("x [cm]")

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
    plt.show()


# ============================================================================
# RAMSES-RT SETUP
# ============================================================================

size_shape = 256
box_width_phys = 1.0          # cm : physical width of the whole box
length_unit_cm = box_width_phys   # 1 code length unit = 1 cm = whole box
cell_size_phys = box_width_phys / size_shape    # cm per cell
cell_size_code = cell_size_phys / length_unit_cm  # = 1/256 code length units

cu = CodeUnits.from_config(
    {
        "length": f"{length_unit_cm} cm",
        "mass": "1 g",
        "velocity": "100000 cm/s",
        "light_speed": "3e10 cm/s",#remove
    },
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

source_rate_phys = "1e10 photons/s"
source_rate_code = to_code(source_rate_phys, "photon_rate", cu)
t_phys = 5.2e-11
time_code = t_phys / cu.T_cgs

print("Snapshot time:", f"{t_phys:.2e} s")

print("\nCode unit scales:")
print(f"  L_cgs = {cu.L_cgs:.6e} cm")
print(f"  V_cgs = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs = {cu.T_cgs:.6e} s")
print(f"  light_speed_code = {cu.light_speed_code:.6e}")

print("\nGrid geometry:")
print(f"  box_width_phys = {box_width_phys:.6e} cm")
print(f"  size_shape = {size_shape}")
print(f"  cell_size_phys = {cell_size_phys:.6e} cm")
print(f"  cell_size_code = {cell_size_code:.6e}")

print("\nCode values:")
print(f"  source_rate_phys = {source_rate_phys}")
print(f"  source_rate_code = {source_rate_code:.4e}")
print(f"  time_code = {time_code:.4e}")

light_cross_time_phys = box_width_phys / 3.0e10
print(f"\nPhysical light-crossing time of box = {light_cross_time_phys:.4e} s")


# ============================================================================
# SIMULATION SETUP
# ============================================================================

eq_test = EquationManager_RT(
    light_speed=cu.light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    debug=False,
)

ss = dh.signal_speed_Rusanov
solver_test = dh.HLL_Radiative_transfer_Local(
    equation_manager=eq_test,
    signal_speed=ss
)

cf_test = dh.ConvectiveFlux_Radiative_transfer(
    eq_test,
    solver_test,
    dh.PLM(limiter="VANLEER"),
    dx=cell_size_code
)

stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=cell_size_code,
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
    n_super_step=10000,
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

# field[0] = photons per cell
sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

field_test, _, _, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test),
    params,
    time_code,
)

print("✓ Simulation completed!")
print(f"  Number of steps: {nombre_de_pas_test_test}")
print(f"  Time step history physical: {float(jnp.sum(dt_historique_test)) * cu.T_cgs:.4e} s")

# ============================================================================
# DIAGNOSTICS
# ============================================================================

dt_total_phys = float(jnp.sum(dt_historique_test)) * cu.T_cgs
N_expected = 1e10 * dt_total_phys
N_gamma_tot = float(np.sum(np.asarray(field_test[0])))

print(f"Expected injected photons = {N_expected:.4e} photons")
print(f"Total photons in domain   = {N_gamma_tot:.4e} photons")
print(f"Difference                = {N_gamma_tot - N_expected:.4e} photons")

extent_xy = [-box_width_phys/2, box_width_phys/2, -box_width_phys/2, box_width_phys/2]

zoom_cells = 12
zoom_half_size = zoom_cells * cell_size_phys

E3d = np.asarray(field_test[0])
E_slice = E3d[:, :, size_shape // 2]

cell_area_phys = cell_size_phys**2
cell_volume_phys = cell_size_phys**3

# photons per cm^3 in the central slice
E_slice_density = E_slice#from_code_fields(E_slice) #/ cell_volume_phys

# projected photons per cm^2
N_col_phys = np.sum(E3d, axis=2) 

# ============================================================================
# PLOTS RAMSES-RT STYLE
# ============================================================================

zoom_half_size = 0.05   # cm : même ordre que la figure RAMSES-RT

# 1) densité volumique en log, proche de la source
make_black_imshow(
    data=E_slice_density,
    extent=extent_xy,
    title=r"Photon number density near source $n_\gamma$ [photons cm$^{-3}$]",
    cbar_label=r"$n_\gamma\ [\mathrm{photons\ cm^{-3}}]$",
    cmap="hot",
    log=True,
    xlim=(-zoom_half_size, zoom_half_size),
    ylim=(-zoom_half_size, zoom_half_size),
)

# 2) densité surfacique projetée en log, proche de la source
make_black_imshow(
    data=N_col_phys,
    extent=extent_xy,
    title=r"Projected photon surface density near source $N$ [photons cm$^{-2}$]",
    cbar_label=r"$N\ [\mathrm{photons\ cm^{-2}}]$",
    cmap="hot",
    log=True,
    xlim=(-zoom_half_size, zoom_half_size),
    ylim=(-zoom_half_size, zoom_half_size),
)