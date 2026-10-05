"""Core corrected Stromgren test runner with CSV, PNG and GIF outputs."""
import os, sys, math, copy as cp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
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
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics import hydrogen_chemistry as hchem

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
N = 50
ULEN = "4.7536191406e22 cm"
BOXPHYS = "4.7536191406e22 cm"
UVEL = "2.99792458e7 cm/s"
SRC = 5e48
TPHYS = "1.92915e16 s"
EPS = 1e-30
NSTEP = 5000
MAXDT = None
N_H_CGS = 1.0e-3
T_AMBIENT_K = 1.0e4
MAKE_GIFS = True
GIF_FRAMES = 25

up = UnitParser()

def parse_quantity(text, expected_dim):
    return up.parse(text, expected_dim=expected_dim)

def sanitize_tag(text):
    return text.replace(" ", "").replace("/", "p").replace("^", "")

# -----------------------------------------------------------------------------
# Units
# -----------------------------------------------------------------------------
size_shape = N
ulen_q = parse_quantity(ULEN, "length")
unit_length_phys_cgs = float(ulen_q.cgs_value)
unit_length_str = f"{ulen_q.value:g}{ulen_q.unit}"
box_q = parse_quantity(BOXPHYS, "length")
box_width_phys_cgs = float(box_q.cgs_value)
box_width_code = box_width_phys_cgs / unit_length_phys_cgs
box_width_str = f"{box_q.value:g}{box_q.unit}"
axis_unit_name = box_q.unit
axis_unit_scale = up.unit_factor_to_cgs(axis_unit_name, expected_dim="length")

dx_code = box_width_code / size_shape
dx_phys_cgs = dx_code * unit_length_phys_cgs
cell_volume_code = dx_code**3
cell_volume_cm3 = dx_phys_cgs**3

uvel_q = parse_quantity(UVEL, "velocity")
unit_velocity_phys = float(uvel_q.cgs_value)
unit_velocity_str = f"{uvel_q.value:g}{uvel_q.unit}"
c_cgs = 2.99792458e10
c_red_cgs = 1.0e-3 * c_cgs
if not np.isclose(unit_velocity_phys, c_red_cgs, rtol=1e-12):
    raise ValueError(f"UVEL must equal reduced light speed {c_red_cgs:.8e} cm/s")

source_rate_phys = float(SRC)
tphys_q = parse_quantity(TPHYS, "time")
t_phys = float(tphys_q.cgs_value)
time_axis_unit = tphys_q.unit
time_axis_scale = up.unit_factor_to_cgs(time_axis_unit, expected_dim="time")
rho_ambient_cgs = N_H_CGS * 1.6726219e-24
mass_unit_phys_cgs = rho_ambient_cgs * unit_length_phys_cgs**3
cu = CodeUnits.from_config(
    {"length": f"{unit_length_phys_cgs} cm", "mass": f"{mass_unit_phys_cgs} g", "velocity": f"{c_red_cgs} cm/s"},
    {"gamma": 5.0 / 3.0, "mu": 1.0},
)

light_speed_code = 1.0
time_code = t_phys / cu.T_cgs
source_rate_code = source_rate_phys * cu.T_cgs
cfl_code = 0.4
dt_cfl = cfl_code / (3.0 * light_speed_code / dx_code)
max_dt = MAXDT if MAXDT is not None else dt_cfl
n_steps_est = int(math.ceil(time_code / dt_cfl))
n_super_step = NSTEP

run_tag = (
    f"N{size_shape}_ulen{sanitize_tag(unit_length_str)}"
    f"_box{sanitize_tag(box_width_str)}_uvel{sanitize_tag(unit_velocity_str)}"
    f"_src{source_rate_phys:.2e}phs_t{sanitize_tag(str(tphys_q.value) + tphys_q.unit)}"
)
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "examples/RT/Images", run_tag)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("Backend:", jax.default_backend(), jax.devices())
print("=" * 70)
for label, value in [
    ("ULEN", f"{unit_length_phys_cgs:.6e} cm"),
    ("BOX", f"{box_width_phys_cgs:.6e} cm"),
    ("UVEL", f"{unit_velocity_phys:.6e} cm/s"),
    ("dx_code", f"{dx_code:.6e}"),
    ("dx_phys_cgs", f"{dx_phys_cgs:.6e} cm"),
    ("L_cgs", f"{cu.L_cgs:.6e} cm"),
    ("V_cgs", f"{cu.V_cgs:.6e} cm/s"),
    ("T_cgs", f"{cu.T_cgs:.6e} s"),
    ("Temp_cgs", f"{cu.Temp_cgs:.6e} K"),
    ("time_code", f"{time_code:.6e}"),
    ("source_rate_code", f"{source_rate_code:.6e}"),
    ("dt_cfl", f"{dt_cfl:.6e}"),
    ("max_dt", f"{max_dt:.6e}"),
]:
    print(f" {label} = {value}")
print(f" estimated n_steps = {n_steps_est} (n_super_step = {n_super_step})")
print("=" * 70)

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
eps_code = float(EPS)
eq_test = EquationManager_RT(light_speed=light_speed_code, mesh_shape=(N, N, N), eps=eps_code, debug=False)
eq_test_hydro = dh.EquationManager(gamma=5.0 / 3.0, n_cons=6, passive_names=("x_HII",), mesh_shape=(N, N, N), eps=eps_code)
assert abs(cfl_code - eq_test.cfl) < 1e-12

class StateBlockFlux:
    def __init__(self, base_flux, state_slice):
        self.base_flux = base_flux
        self.state_slice = state_slice
        self.dx_o = base_flux.dx_o
    def flux(self, sol, ax, params, flux):
        local = self.base_flux.flux(sol[self.state_slice], ax, params, flux)
        return jnp.zeros_like(sol).at[self.state_slice].set(local)
    def timestep(self, sol):
        return self.base_flux.timestep(sol[self.state_slice])

rt_flux = StateBlockFlux(dh.ConvectiveFlux_Radiative_transfer(eq_test, dh.HLL_Radiative_transfer_Local(eq_test, dh.signal_speed_Rusanov), dh.PLM(limiter="MINMOD"), dx=dx_code), slice(0, eq_test.n_cons))
hydro_flux = StateBlockFlux(dh.ConvectiveFlux(eq_test_hydro, dh.LaxFriedrichs(eq_test_hydro, dh.signal_speed_Rusanov), dh.PLM(limiter="MINMOD"), dx=dx_code), slice(eq_test.n_cons, eq_test.n_cons + eq_test_hydro.n_cons))

stellar_force = StellarRadiationForce(dx=dx_code, injection_mode="stromgren", stromgren_rate=source_rate_code, injection_momentum=True, injection_geometry="radial_3D", gaussian_star=True, beam_momentum_scaling="legacy_c2_source2", beam_reduced_flux=0.75, eq=eq_test, hydro_eq=eq_test_hydro, cu=cu, chemistry=False)
chem_force = HydrogenPhotoChemistryForce(stellar_force, case="B", collisional=False, max_frac=0.9, include_heating=False, include_cooling=False)
hydrosim_test = dh.hydro(n_super_step=n_super_step, fluxes=[hydro_flux, rt_flux], forces=[stellar_force,chem_force], dx=dx_code, max_dt=max_dt)
assert hydrosim_test.dx_o == rt_flux.dx_o == stellar_force.dx

# -----------------------------------------------------------------------------
# Initial condition and main run
# -----------------------------------------------------------------------------
params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.1]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[N // 2] * 3], dtype=jnp.int32),
}
n_total_fields = eq_test.n_cons + eq_test_hydro.n_cons
sol_test = jnp.zeros((n_total_fields, N, N, N), dtype=jnp.float64)
center = N // 2
idx_rho_local = eq_test.n_cons + eq_test_hydro.mass_ids
idx_p_local = eq_test.n_cons + eq_test_hydro.energy_ids
mH_cgs = 1.6726219e-24
kB_cgs = 1.380649e-16
rho_ambient_code = rho_ambient_cgs / cu.rho_cgs
p_ambient_code = rho_ambient_cgs * kB_cgs * T_AMBIENT_K / mH_cgs / cu.P_cgs
sol_test = sol_test.at[idx_rho_local].set(rho_ambient_code)
sol_test = sol_test.at[idx_p_local].set(p_ambient_code)
sol_test = sol_test.at[0, center, center, center].set(1e-20)
sol_test = sol_test.at[9].set(0.0)

print(f"Running to t = {t_phys:.6e} s = {time_code:.6e} code")
field_test, params_final, _, dt_hist, n_steps = hydrosim_test.evolve_till_time(cp.deepcopy(sol_test), params, time_code)
field_test = field_test.at[0].set(jnp.maximum(field_test[0], 0.0))
dt_hist = np.asarray(dt_hist)
dt_sum = float(dt_hist[dt_hist > 0].sum())
E3d = np.asarray(field_test[0], dtype=np.float64)


Fx3d = np.asarray(field_test[1], dtype=np.float64)
Fy3d = np.asarray(field_test[2], dtype=np.float64)
Fz3d = np.asarray(field_test[3], dtype=np.float64)

Fmag = np.sqrt(Fx3d**2 + Fy3d**2 + Fz3d**2)
E_safe = np.maximum(E3d, 1e-300)

reduced_flux = Fmag / (light_speed_code * E_safe)

print(
    "DEBUG reduced flux min/max/mean =",
    float(np.nanmin(reduced_flux)),
    float(np.nanmax(reduced_flux)),
    float(np.nanmean(reduced_flux)),
)

axis_line = reduced_flux[center:, center, center]
diag_line = np.array([
    reduced_flux[center + i, center + i, center]
    for i in range(min(center, N - center))
])

print(
    "DEBUG reduced flux axis/diag max =",
    float(np.nanmax(axis_line)),
    float(np.nanmax(diag_line)),
)

Fx3d = np.asarray(field_test[1], dtype=np.float64)
Fy3d = np.asarray(field_test[2], dtype=np.float64)
Fz3d = np.asarray(field_test[3], dtype=np.float64)

xxc = np.arange(N) - center
yyc = np.arange(N) - center
zzc = np.arange(N) - center
gx, gy, gz = np.meshgrid(xxc, yyc, zzc, indexing="ij")

rr = np.sqrt(gx**2 + gy**2 + gz**2)
rr_safe = np.maximum(rr, 1.0)

Fr = (Fx3d * gx + Fy3d * gy + Fz3d * gz) / rr_safe
Ft2 = np.maximum(Fx3d**2 + Fy3d**2 + Fz3d**2 - Fr**2, 0.0)

mask = (rr > 2.0) & (E3d > E3d.max() * 1e-10)

print(
    "DEBUG radial flux fraction mean/max =",
    float(np.mean(Fr[mask] / np.maximum(np.sqrt(Fx3d[mask]**2 + Fy3d[mask]**2 + Fz3d[mask]**2), 1e-300))),
    float(np.max(Fr[mask] / np.maximum(np.sqrt(Fx3d[mask]**2 + Fy3d[mask]**2 + Fz3d[mask]**2), 1e-300))),
)

print(
    "DEBUG transverse flux fraction mean/max =",
    float(np.mean(np.sqrt(Ft2[mask]) / np.maximum(np.sqrt(Fx3d[mask]**2 + Fy3d[mask]**2 + Fz3d[mask]**2), 1e-300))),
    float(np.max(np.sqrt(Ft2[mask]) / np.maximum(np.sqrt(Fx3d[mask]**2 + Fy3d[mask]**2 + Fz3d[mask]**2), 1e-300))),
)

E_cell = E3d * cell_volume_code
xHII_abs_idx = getattr(stellar_force, "idx_xHII", None)
if xHII_abs_idx is None or field_test.shape[0] <= xHII_abs_idx:
    raise RuntimeError("x_HII field is unavailable")
xHII_3d = np.asarray(field_test[xHII_abs_idx], dtype=np.float64)

print("Done.")
print(f" steps = {int(n_steps)}")
print(f" sum(dt) = {dt_sum:.6e} code = {dt_sum * cu.T_cgs:.6e} s")
print(f" target = {time_code:.6e} code")
print(f" E_gamma min/max/negative = {E3d.min():.6e} / {E3d.max():.6e} / {int(np.sum(E3d < 0.0))}")
photons_in_box = float(E_cell.sum())
photons_expected = float(source_rate_code * dt_sum)
print(f" photons in box = {photons_in_box:.6e}")
print(f" photons injected = {photons_expected:.6e}")
print(f" radiation-only ratio = {photons_in_box / max(photons_expected, 1e-300):.6e}")
print(f" x_HII min/max/mean = {xHII_3d.min():.6e} / {xHII_3d.max():.6e} / {xHII_3d.mean():.6e}")

# -----------------------------------------------------------------------------
# Analytic solution and front
# -----------------------------------------------------------------------------
alpha_B_cgs = float(hchem.alpha_B_HII_cgs(T_AMBIENT_K))
t_rec_cgs = 1.0 / (alpha_B_cgs * N_H_CGS)
R_stromgren_cgs = (3.0 * source_rate_phys / (4.0 * np.pi * alpha_B_cgs * N_H_CGS**2)) ** (1.0 / 3.0)
R_I_t_cgs = R_stromgren_cgs * (-np.expm1(-t_phys / t_rec_cgs)) ** (1.0 / 3.0)

xx, yy, zz = np.meshgrid(np.arange(N) - center, np.arange(N) - center, np.arange(N) - center, indexing="ij")
r_cells = np.sqrt(xx**2 + yy**2 + zz**2)
r_int = np.round(r_cells).astype(int)
r_vals = np.arange(r_int.max() + 1)
x_shell = np.array([np.mean(xHII_3d[r_int == r]) if np.any(r_int == r) else np.nan for r in r_vals])
above = np.where(x_shell >= 0.5)[0]
if above.size:
    i1 = above[-1]
    if i1 >= len(r_vals) - 1:
        r_front_cells = float(r_vals[i1])
    else:
        r1, r2 = float(r_vals[i1]), float(r_vals[i1 + 1])
        x1, x2 = float(x_shell[i1]), float(x_shell[i1 + 1])
        r_front_cells = r1 if abs(x2 - x1) < 1e-14 else r1 + (0.5 - x1) * (r2 - r1) / (x2 - x1)
    r_front_cgs = r_front_cells * dx_phys_cgs
else:
    r_front_cells = np.nan
    r_front_cgs = np.nan

print("=" * 70)
print(f" alpha_B = {alpha_B_cgs:.6e} cm^3 s^-1")
print(f" R_stromgren = {R_stromgren_cgs:.6e} cm")
print(f" R_I(t) = {R_I_t_cgs:.6e} cm")
print(f" simulated interpolated front = {r_front_cells:.6f} cells = {r_front_cgs:.6e} cm")
print(f" ratio simulated / analytic = {r_front_cgs / R_I_t_cgs:.6f}")

rho_cgs_3d = np.asarray(field_test[idx_rho_local], dtype=np.float64) * cu.rho_cgs
nH_3d = rho_cgs_3d / mH_cgs
ionized_atoms = float(np.sum(nH_3d * xHII_3d) * cell_volume_cm3)
print(f" ionized H atoms = {ionized_atoms:.6e}")
print(f" injected photons / ionized atoms = {photons_expected / max(ionized_atoms, 1e-300):.6e}")

# -----------------------------------------------------------------------------
# Output directories and plots
# -----------------------------------------------------------------------------
out_fields = os.path.join(BASE_OUTPUT_DIR, "fields")
os.makedirs(out_fields, exist_ok=True)
extent = [0.0, N * dx_phys_cgs / axis_unit_scale, 0.0, N * dx_phys_cgs / axis_unit_scale]
zslice = center

def save_image(data, path, title, label, cmap="viridis", vmin=None, vmax=None, norm=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xlabel(f"x [{axis_unit_name}]")
    ax.set_ylabel(f"y [{axis_unit_name}]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Main figures
E_slice = E_cell[:, :, zslice]
positive = E_slice[E_slice > 0]
if positive.size:
    save_image(E_slice, os.path.join(BASE_OUTPUT_DIR, f"field_test_photons_per_cell_{run_tag}.png"), f"Photons per cell, t={t_phys:.3e} s", "photons/cell", cmap="hot", norm=LogNorm(vmin=max(float(positive.min()), float(E_slice.max()) * 1e-12), vmax=float(positive.max())))
    save_image(E3d[:, :, zslice], os.path.join(BASE_OUTPUT_DIR, f"field_test_Egamma_code_{run_tag}.png"), f"E_gamma code, t={t_phys:.3e} s", "photons/code volume", cmap="hot")
    log_slice = np.log10(np.maximum(E3d[:, :, zslice], 1e-100))
    save_image(log_slice, os.path.join(BASE_OUTPUT_DIR, f"field_test_Egamma_log_{run_tag}.png"), f"log10 E_gamma, t={t_phys:.3e} s", "log10 photons/code volume", cmap="hot")

save_image(xHII_3d[:, :, zslice], os.path.join(BASE_OUTPUT_DIR, f"xHII_slice_{run_tag}.png"), f"Ionization fraction x_HII, t={t_phys:.3e} s", "x_HII", cmap="viridis", vmin=0.0, vmax=1.0)

# Radial front figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_vals * dx_phys_cgs / axis_unit_scale, x_shell, "o-", ms=3, label="simulated shell average")
ax.axvline(R_I_t_cgs / axis_unit_scale, color="orange", ls="--", label="analytic R_I(t)")
ax.axhline(0.5, color="gray", ls=":")
ax.set_xlabel(f"r [{axis_unit_name}]")
ax.set_ylabel("shell-average x_HII")
ax.set_title("Stromgren sphere validation")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(BASE_OUTPUT_DIR, f"stromgren_test_{run_tag}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Per-field PNG and CSV files
field_names = ["E_gamma", "Fx", "Fy", "Fz", "rho", "vx", "vy", "vz", "p", "x_HII"]
assert field_test.shape[0] == len(field_names)
for k, name in enumerate(field_names):
    field_dir = os.path.join(out_fields, name)
    os.makedirs(field_dir, exist_ok=True)
    arr = np.asarray(field_test[k], dtype=np.float64)
    save_image(arr[:, :, zslice], os.path.join(field_dir, f"slice_{name}_{run_tag}.png"), f"{name} slice, z={zslice}", name, cmap="viridis")
    log_arr = np.log10(np.maximum(np.abs(arr), 1e-100))
    save_image(log_arr[:, :, zslice], os.path.join(field_dir, f"slice_{name}_log_{run_tag}.png"), f"log10 |{name}| slice, z={zslice}", f"log10 |{name}|", cmap="magma")
    coords = np.indices(arr.shape, dtype=np.int32)
    csv_path = os.path.join(field_dir, f"cube_{name}_{run_tag}.csv")
    np.savetxt(csv_path, np.column_stack([coords[0].ravel(), coords[1].ravel(), coords[2].ravel(), arr.ravel()]), delimiter=",", header=f"i,j,k,{name}", comments="", fmt=["%d", "%d", "%d", "%.8e"])

# Radial profile CSV
np.savetxt(os.path.join(BASE_OUTPUT_DIR, f"radial_xHII_{run_tag}.csv"), np.column_stack([r_vals, r_vals * dx_phys_cgs, x_shell]), delimiter=",", header="radius_cells,radius_cm,xHII_shell_average", comments="", fmt="%.8e")

# Optional GIFs: frame 0 plus incremental runs
if MAKE_GIFS:
    gif_dir = os.path.join(BASE_OUTPUT_DIR, "field_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    dt_frame_code = time_code / GIF_FRAMES
    gif_names = ["E_gamma", "Fx", "Fy", "Fz", "rho", "vx", "vy", "vz", "p", "x_HII"]
    sol_current = cp.deepcopy(sol_test)
    frames = {name: [np.asarray(sol_current[k, :, :, zslice], dtype=np.float64)] for k, name in enumerate(gif_names)}
    frame_times = [0.0]
    t_accum = 0.0
    for frame_i in range(GIF_FRAMES):
        sol_current, _, _, _, _ = hydrosim_test.evolve_till_time(sol_current, params, dt_frame_code)
        t_accum += dt_frame_code * cu.T_cgs
        frame_times.append(t_accum)
        for k, name in enumerate(gif_names):
            frames[name].append(np.asarray(sol_current[k, :, :, zslice], dtype=np.float64))
    for name in gif_names:
        stack = np.stack(frames[name])
        if name == "x_HII":
            cmap, vmin, vmax = "viridis", 0.0, 1.0
        elif name in {"Fx", "Fy", "Fz", "vx", "vy", "vz"}:
            cmap = "coolwarm"
            a = float(np.max(np.abs(stack)))
            vmin, vmax = -a, a
        else:
            cmap = "hot"
            vmin, vmax = 0.0, float(np.max(stack))
        frame_paths = []
        frame_dir = os.path.join(gif_dir, f"{name}_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(stack):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(frame, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel(f"x [{axis_unit_name}]")
            ax.set_ylabel(f"y [{axis_unit_name}]")
            ax.set_title(f"{name}, t={frame_times[i]:.3e} s")
            fig.colorbar(im, ax=ax, label=name)
            path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(path)
        images = [Image.open(p) for p in frame_paths]
        gif_path = os.path.join(gif_dir, f"{name}_evolution_{run_tag}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=[800] + [200] * (len(images) - 1), loop=0)
        for image in images:
            image.close()

print(f"Outputs written to: {BASE_OUTPUT_DIR}")
