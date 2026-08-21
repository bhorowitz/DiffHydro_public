"""Updated Stromgren validation runner with the validated RT configuration."""
import math
import os
import sys
import copy as cp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import EquationManager as EquationManagerRT
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits

jax.config.update("jax_enable_x64", True)
print("Backend:", jax.default_backend())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", 50))
n_H_cgs = 1.0e-3
T_K = 1.0e4
Q_phot = 5.0e48
GAMMA = 5.0 / 3.0
RSLA = float(os.environ.get("RSLA", 2.0e-2))
TEND_REC = float(os.environ.get("TEND", 5.0))
CFL = 0.4
NSTEP = int(os.environ.get("NSTEP", 5000))
MAKE_GIFS = os.environ.get("MAKE_GIFS", "0") == "1"
GIF_FRAMES = int(os.environ.get("GIF_FRAMES", 25))
LIMITER = "MINMOD"
FMAX = 0.75

alpha_B = float(hchem.alpha_B_HII_cgs(T_K))
R_S = (3.0 * Q_phot / (4.0 * np.pi * alpha_B * n_H_cgs**2)) ** (1.0 / 3.0)
t_rec = 1.0 / (alpha_B * n_H_cgs)
box_cgs = 4.0 * R_S / 1.4
dx_cgs = box_cgs / N
t_end = TEND_REC * t_rec
c_red_cgs = RSLA * hchem.C_LIGHT_CGS

M_unit_cgs = n_H_cgs * hchem.MH_CGS * dx_cgs**3
cu = CodeUnits.from_config(
    {"length": f"{dx_cgs} cm", "mass": f"{M_unit_cgs} g", "velocity": f"{c_red_cgs} cm/s"},
    {"gamma": GAMMA, "mu": 1.0},
)

dx_code = 1.0
c_code = 1.0
dt_code = CFL / (3.0 * c_code / dx_code)
time_code = t_end / cu.T_cgs
n_steps_est = int(math.ceil(time_code / dt_code))
n_super_step = max(NSTEP, n_steps_est + 10)

print("=" * 72)
print(f" N={N}, dx={dx_cgs:.6e} cm, box={box_cgs:.6e} cm")
print(f" Q={Q_phot:.6e} ph/s, alpha_B={alpha_B:.6e} cm^3/s")
print(f" R_S={R_S:.6e} cm, t_rec={t_rec:.6e} s")
print(f" RSLA={RSLA:.6e}, c_red={c_red_cgs:.6e} cm/s")
print(f" time_code={time_code:.6e}, dt_code={dt_code:.6e}")
print(f" estimated n_steps={n_steps_est}, n_super_step={n_super_step}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
eq_rt = EquationManagerRT(light_speed=c_code, mesh_shape=(N, N, N), eps=1e-30, debug=False)
eq_hydro = dh.EquationManager(gamma=GAMMA, n_cons=6, passive_names=("x_HII",), mesh_shape=(N, N, N), eps=1e-30)

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

rt_flux = StateBlockFlux(
    dh.ConvectiveFlux_Radiative_transfer(
        eq_rt,
        dh.HLL_Radiative_transfer_Local(eq_rt, dh.signal_speed_Rusanov),
        dh.PLM(limiter=LIMITER),
        dx=dx_code,
    ),
    slice(0, eq_rt.n_cons),
)
hydro_flux = StateBlockFlux(
    dh.ConvectiveFlux(
        eq_hydro,
        dh.LaxFriedrichs(eq_hydro, dh.signal_speed_Rusanov),
        dh.PLM(limiter=LIMITER),
        dx=dx_code,
    ),
    slice(eq_rt.n_cons, eq_rt.n_cons + eq_hydro.n_cons),
)

stellar = StellarRadiationForce(
    dx=dx_code,
    injection_mode="stromgren",
    stromgren_rate=Q_phot * cu.T_cgs,
    injection_momentum=True,
    injection_geometry="radial_3D",
    gaussian_star=True,
    beam_momentum_scaling="legacy_c2_source2",
    beam_reduced_flux=FMAX,
    eq=eq_rt,
    hydro_eq=eq_hydro,
    cu=cu,
    chemistry=False,
)
chem_force = HydrogenPhotoChemistryForce(
    stellar,
    case="B",
    collisional=False,
    max_frac=0.9,
    include_heating=False,
    include_cooling=False,
)

sim = dh.hydro(
    n_super_step=n_super_step,
    fluxes=[hydro_flux, rt_flux],
    forces=[stellar, chem_force],
    dx=dx_code,
    max_dt=dt_code,
)

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
rho_code = n_H_cgs * hchem.MH_CGS / cu.rho_cgs
p_code = n_H_cgs * hchem.KB_CGS * T_K / cu.P_cgs
sol = jnp.zeros((10, N, N, N), dtype=jnp.float64)
sol = sol.at[4].set(rho_code)
sol = sol.at[8].set(p_code / (GAMMA - 1.0))
center = N // 2
sol = sol.at[9].set(0.0)

params = {
    "star_masses": jnp.array([1.0]),
    "star_ages": jnp.array([0.0]),
    "star_metallicities": jnp.array([0.02]),
    "star_positions": jnp.array([[center] * 3], dtype=jnp.int32),
}

# ---------------------------------------------------------------------------
# Snapshot evolution
# ---------------------------------------------------------------------------
n_snap = 12
snap_every = max(1, n_steps_est // n_snap)

def xHII_from_conservative(state):
    return np.asarray(chem_force.view.xHII(state), dtype=np.float64)

def ionized_radius(x3d):
    V = float(np.sum(np.asarray(x3d, dtype=np.float64))) * dx_cgs**3
    return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)

@jax.jit
def run_chunk(state, pars, step0):
    def body(j, carry):
        s, p = carry
        return sim._hydrostep(step0 + j, (s, p), dt_code)
    return jax.lax.fori_loop(0, snap_every, body, (state, pars))

times = [0.0]
radii = [0.0]
snap_states = [np.asarray(sol)]
t_code = 0.0
n_chunks = int(math.ceil(n_steps_est / snap_every))
for chunk in range(n_chunks):
    steps_this = min(snap_every, n_steps_est - chunk * snap_every)
    if steps_this != snap_every:
        def body(j, carry):
            s, p = carry
            return sim._hydrostep(chunk * snap_every + j, (s, p), dt_code)
        sol, params = jax.lax.fori_loop(0, steps_this, body, (sol, params))
    else:
        sol, params = run_chunk(sol, params, chunk * snap_every)
    t_code += steps_this * dt_code
    x3d = xHII_from_conservative(sol)
    t_s = t_code * cu.T_cgs
    times.append(t_s)
    radii.append(ionized_radius(x3d))
    snap_states.append(np.asarray(sol))
    print(f" step {min((chunk + 1) * snap_every, n_steps_est):5d}/{n_steps_est} "
          f"t/t_rec={t_s/t_rec:6.3f} R/R_S={radii[-1]/R_S:7.4f} "
          f"x_max={x3d.max():.4f}")

times = np.asarray(times)
radii = np.asarray(radii)
analytic = R_S * (1.0 - np.exp(-times / t_rec)) ** (1.0 / 3.0)
rel_err = np.abs(radii[1:] - analytic[1:]) / np.maximum(analytic[1:], 1e-300)
print("=" * 72)
print(f" final R_I={radii[-1]:.6e} cm, analytic={analytic[-1]:.6e} cm, ratio={radii[-1]/analytic[-1]:.6f}")
print(f" mean relative error={rel_err.mean()*100:.3f}%")
print("=" * 72)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
out_dir = os.path.join(REPO_ROOT, "examples/RT/Images/stromgren_validation_updated")
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, f"history_N{N}.csv"), np.column_stack([times, radii, analytic, radii / np.maximum(analytic, 1e-300)]), delimiter=",", header="time_s,radius_cm,analytic_radius_cm,ratio", comments="")

x_final = xHII_from_conservative(sol)
np.savetxt(os.path.join(out_dir, f"radial_profile_N{N}.csv"), np.column_stack([np.arange(N), np.nanmean(x_final, axis=(1, 2))]), delimiter=",", header="index,xHII_mean", comments="")

# Main validation figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
t_fine = np.linspace(0, times[-1], 400)
axes[0].plot(t_fine / t_rec, (1.0 - np.exp(-t_fine / t_rec)) ** (1.0 / 3.0), "k-", lw=2, label="analytic")
axes[0].plot(times / t_rec, radii / R_S, "o--", color="tab:red", ms=5, label=f"DiffHydro N={N}")
axes[0].set(xlabel="t/t_rec", ylabel="R_I/R_S", title="R-type HII expansion")
axes[0].grid(alpha=0.3)
axes[0].legend()

x_slice = x_final[:, :, center]
extent_pc = [-0.5 * box_cgs / 3.0857e18, 0.5 * box_cgs / 3.0857e18] * 2
im = axes[1].imshow(x_slice.T, origin="lower", cmap="magma", vmin=0, vmax=1, extent=extent_pc)
th = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(R_S / 3.0857e18 * np.cos(th), R_S / 3.0857e18 * np.sin(th), "c--", lw=1.5, label="R_S")
axes[1].set(xlabel="x [pc]", ylabel="y [pc]", title=fr"$x_{{HII}}$, t={times[-1]/t_rec:.2f} t_rec")
axes[1].legend()
fig.colorbar(im, ax=axes[1], label="x_HII")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"stromgren_N{N}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Final-field PNGs and CSV cubes
field_names = ["E_gamma", "Fx", "Fy", "Fz", "rho", "vx", "vy", "vz", "p", "x_HII"]
field_root = os.path.join(out_dir, f"fields_N{N}")
os.makedirs(field_root, exist_ok=True)
extent_code = [0, N, 0, N]
for k, name in enumerate(field_names):
    arr = np.asarray(sol[k], dtype=np.float64)
    field_dir = os.path.join(field_root, name)
    os.makedirs(field_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr[:, :, center].T, origin="lower", cmap="viridis", extent=extent_code)
    ax.set_title(f"{name}, final")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax, label=name)
    fig.tight_layout()
    fig.savefig(os.path.join(field_dir, f"slice_{name}_N{N}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    ii, jj, kk = np.indices(arr.shape)
    np.savetxt(os.path.join(field_dir, f"cube_{name}_N{N}.csv"), np.column_stack([ii.ravel(), jj.ravel(), kk.ravel(), arr.ravel()]), delimiter=",", header=f"i,j,k,{name}", comments="", fmt=["%d", "%d", "%d", "%.8e"])

# Optional GIFs from stored snapshots
if MAKE_GIFS:
    gif_root = os.path.join(out_dir, f"gifs_N{N}")
    os.makedirs(gif_root, exist_ok=True)
    gif_names = field_names
    for k, name in enumerate(gif_names):
        stack = np.stack([snap[k, :, :, center] for snap in snap_states])
        if name == "x_HII":
            cmap, vmin, vmax = "viridis", 0.0, 1.0
        elif name in {"Fx", "Fy", "Fz", "vx", "vy", "vz"}:
            cmap = "coolwarm"
            a = float(np.max(np.abs(stack)))
            vmin, vmax = -a, a
        else:
            cmap = "hot"
            vmin, vmax = float(np.min(stack)), float(np.max(stack))
        frame_dir = os.path.join(gif_root, f"{name}_frames")
        os.makedirs(frame_dir, exist_ok=True)
        paths = []
        for i, frame in enumerate(stack):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(frame.T, origin="lower", cmap=cmap, extent=extent_code, vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}, t={times[i]/t_rec:.3f} t_rec")
            ax.set_xlabel("x cell")
            ax.set_ylabel("y cell")
            fig.colorbar(im, ax=ax, label=name)
            fig.tight_layout()
            path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        images = [Image.open(p) for p in paths]
        gif_path = os.path.join(gif_root, f"{name}_N{N}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=[700] + [250] * (len(images) - 1), loop=0)
        for image in images:
            image.close()

print(f"Outputs written to {out_dir}")
