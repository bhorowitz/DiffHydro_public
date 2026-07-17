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

from diffhydro.units import CodeUnits, to_code
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
                      xlim=None, ylim=None, figsize=(6, 5)):
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
    plt.savefig(f"examples/athena/Images_athena/{title}_c=1e5.png", dpi=300, bbox_inches="tight")
    plt.show()



# ============================================================================
# RAMSES-RT SETUP
# ============================================================================

size_shape = 256
box_width_phys = 1.0  # cm
unit_length = 1  # cm
dx_cell_code = box_width_phys / size_shape  # code units
# dx_phys = 1.0 / size_shape   # cm per cell

cu = CodeUnits.from_config(
    {
        "length": f"{unit_length} cm",
        "mass": "1 g",
        "velocity": "3e10 cm/s",
    },
    {"gamma": 5.0 / 3.0, "mu": 0.61},
)

source_rate_phys = "1e10 photons/s"
source_rate_code = to_code(source_rate_phys, "photon_rate", cu)
print(f"Source rate: {source_rate_phys} = {source_rate_code:.4e} code units")
t_phys = 5.2e-11
time_code = to_code(f"{t_phys} s", "time", cu)
print("INFORMATIONS : ===============================")
print("time_code =", time_code)
print("box_width_phys =", box_width_phys)
print("dx_cell_code =", dx_cell_code)
print("light_speed_code =", cu.light_speed_code)

# print("=" * 70)
# print("RAMSES-RT UNIT SYSTEM")
# print("=" * 70)
# print("\nPhysical domain:")
# print("  Size: 1 cm (box width in paper)")
# print("  Resolution: 256² cells (paper)")
# print("  Source: 10¹⁰ photons/s (isotropic point source)")
print("  Snapshot time:", f"{t_phys:.2e} s")

print("\nCode unit scales:")
print(f"  L_cgs = {cu.L_cgs:.6e} cm")
print(f"  V_cgs = {cu.V_cgs:.6e} cm/s")
print(f"  T_cgs = {cu.T_cgs:.6e} s")
print(f"  c_cgs = {cu.light_speed_cgs:.6e} cm/s")
print(f"  light_speed_code = {cu.light_speed_code:.6e}")
print(f"  dx_code = {dx_cell_code:.6e}")
print(f"  dt_light_crossing_code = {dx_cell_code / cu.light_speed_code:.6e}")

print("\nCode values:")
print(f"  source_rate_phys = {source_rate_phys}")
print(f"  source_rate_code = {source_rate_code:.4e}")
print(f"  time_code = {time_code:.4e}")
box_crossing_time_code = (box_width_phys / cu.L_cgs) / cu.light_speed_code
print(f"  box_crossing_time_code = {box_crossing_time_code:.4e}")
print(f"  box_crossing_times = {time_code / box_crossing_time_code:.2f}")

light_cross_time_phys = cu.L_cgs / 3.0e10
print(f"\nPhysical light-crossing time of box = {light_cross_time_phys:.4e} s")


# ============================================================================
# SIMULATION SETUP
# ============================================================================


eq_test = EquationManager_RT(
    light_speed=cu.light_speed_code,
    mesh_shape=(size_shape, size_shape, size_shape),
    debug=False,
)

print("=" * 70)
print("RAMSES-RT SIMULATION SETUP")
print("=" * 70)

print(eq_test)
print(f"\n✓ Resolution: {size_shape}³ cells")
print(f"✓ Light speed: {cu.light_speed_code:.4e} code units")
# print("✓ Domain size: 1 cm (in code units = 1.0)")

ss = dh.signal_speed_Rusanov

solver_test = dh.HLL_Radiative_transfer_Local(equation_manager=eq_test, signal_speed=ss)


cf_test = dh.ConvectiveFlux_Radiative_transfer(eq_test, solver_test, dh.PLM(limiter="VANLEER"),dx=dx_cell_code)

stellar_force = StellarRadiationForce(
    escape_fraction=0.1,
    dx=dx_cell_code,
    injection_mode="stromgren",
    stromgren_rate=source_rate_code,
    injection_momentum=False,
    gaussian_star=False,
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

print("hydrosim_test.dxo =", hydrosim_test.dx_o)
# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array(
        [[size_shape // 2, size_shape // 2, size_shape // 2]],
        dtype=jnp.int32,
    ),
}

# Convention B:
# field[0] = photons per cell
sol_test = jnp.zeros((4, size_shape, size_shape, size_shape))

print("=" * 70)
print("RAMSES-RT SIMULATION RUN")
print("=" * 70)
print("\nInitial conditions:")
print(f"  Domain size: {size_shape}³ cells")
print("  Point source: isotropic at center")
print(f"  Source strength: {source_rate_phys} = {source_rate_code:.4e} code units")

print(f"\nRunning simulation until t = {t_phys:.2e} s = {time_code:.4e} code units...")

field_test, _, _, dt_historique_test, nombre_de_pas_test_test = hydrosim_test.evolve_till_time(
    cp.deepcopy(sol_test),
    params,
    time_code,
)

print("✓ Simulation completed!")
print(f"  E_gamma(cell photons): min={float(jnp.min(field_test[0])):.4e}, max={float(jnp.max(field_test[0])):.4e}")
print(f"  |F|_max: {float(jnp.sqrt(jnp.sum(field_test[1:]**2, axis=0)).max()):.4e}")
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

# Radius estimate from projected map
dx_phys = cu.L_cgs / size_shape
half_box = cu.L_cgs / 2.0
extent_xy = [-box_width_phys/2, box_width_phys/2, -box_width_phys/2, box_width_phys/2]
zoom_half_size = 12 * dx_phys

E3d = np.asarray(field_test[0])
E_slice = E3d[:, :, size_shape // 2]

cell_area_phys = dx_phys**2
N_col_phys = np.sum(E3d, axis=2) / cell_area_phys

positive_col = N_col_phys[N_col_phys > 0]
if positive_col.size:
    peak_col = positive_col.max()
    yy, xx = np.indices(N_col_phys.shape)
    cx = cy = size_shape // 2
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2) * dx_phys

    for th_frac in [1e-3, 1e-6, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30]:
        threshold = peak_col * th_frac
        mask = N_col_phys > threshold
        radius_est = rr[mask].max() if np.any(mask) else 0.0
        print(f"threshold={th_frac:.0e} of peak -> radius_est = {radius_est:.4e} cm ({radius_est/dx_phys:.1f} cells)")


r_expected = 3.0e10 * t_phys
print(f"Free-streaming radius c*t = {r_expected:.4e} cm")
print(f"Free-streaming radius in cells = {r_expected / dx_phys:.2f}")


# ============================================================================
# PLOTS
# ============================================================================
print(dt_historique_test, nombre_de_pas_test_test)
zoom_cells = 12
zoom_half_size = zoom_cells * dx_phys

# Full-box and local views help separate real transport from the symmetric injection core.
make_black_imshow(
    data=E_slice,
    extent=extent_xy,
    title="Photon_number_per_cell_full_box_(log scale)",
    cbar_label="Photons per cell",
    cmap="hot",
    log=False,
)

# Linear center slice
make_black_imshow(
    data=E_slice,
    extent=extent_xy,
    title="Photon_number_per_cell_at_box_center",
    cbar_label="Photons per cell",
    cmap="hot",
    log=False,
)

# Zoom linear
make_black_imshow(
    data=E_slice,
    extent=extent_xy,
    title="Photon_number_per_cell_near_source",
    cbar_label="Photons per cell",
    cmap="hot",
    log=False,
    xlim=(-zoom_half_size, zoom_half_size),
    ylim=(-zoom_half_size, zoom_half_size),
)

# Log center slice
make_black_imshow(
    data=np.log(E_slice),
    extent=extent_xy,
    title="Photon_number_per_cell_at_box_center_(log scale)",
    cbar_label="Photons per cell",
    cmap="hot",
    log=True,
)

# Log zoom
make_black_imshow(
    data=E_slice,
    extent=extent_xy,
    title="Photon_number_per_cell_near_source_(log scale)",
    cbar_label="Photons per cell",
    cmap="hot",
    log=True,
    xlim=(-zoom_half_size, zoom_half_size),
    ylim=(-zoom_half_size, zoom_half_size),
)

# Projection map
make_black_imshow(
    data=N_col_phys,
    extent=extent_xy,
    title=r"Projected_photon_surface_density_$N$ [photons cm$^{-3}$]",
    cbar_label=r"$N\ [\mathrm{photons\ cm^{-3}}]$",
    cmap="hot",
    log=True,
)

# Projection zoom
make_black_imshow(
    data=N_col_phys,
    extent=extent_xy,
    title=r"Projected_photon_surface_density_near_source_$N$ [photons cm$^{-3}$]",
    cbar_label=r"$N\ [\mathrm{photons\ cm^{-3}}]$",
    cmap="hot",
    log=True,
    xlim=(-zoom_half_size, zoom_half_size),
    ylim=(-zoom_half_size, zoom_half_size),
)
plt.imshow(np.log(field_test[0, :,:,256//2]), origin="lower", cmap="hot", vmin=-20)
# plt.title(f"field_test[0] at center point ({ix_center}, {iy_center}, {iz_center}), injection {injection_density} ")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.savefig(f"examples/athena/Images_athena/field_test[0]_at_center_point_c=1.png", dpi=300, bbox_inches="tight")
plt.show()