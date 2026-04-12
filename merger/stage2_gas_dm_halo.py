"""
stage2_gas_dm_halo.py
=====================
Stage 2: Single-halo gas+DM, non-cosmological DiffHydro run.

Uses the same cluster model as the GAMER single-cluster static test
(M200=5e14 Msun, conc=3.5, z=0.295, Vikhlinin gas + sNFW DM, 64^3, 6144 kpc box).

What this script does:
  1. Read the cluster_generator profile from gamer_single_cluster/profile1.h5
  2. Map 1D gas density + HSE pressure profiles onto the 3D DiffHydro grid
  3. Map 1D DM density (= total_density - gas_density) onto a static 3D field
     used as the gravitational source — this avoids PM-softening issues that
     arise when CIC-depositing particles with Δx=96 kpc (the particle softening
     length equals the cell size, severely under-resolving the cluster core where
     the steep Vikhlinin pressure gradient needs to be balanced by gravity).
  4. Gas self-gravity is added live via include_gas_in_gravity=True, so the
     Poisson source is rho_DM_analytic + rho_gas ≈ rho_total — exactly the total
     density that cluster_generator used to integrate the HSE pressure profile.
  5. Produce radial gas density and pressure profiles at t=0 and t=final vs target,
     plus xy density slices for direct comparison with the GAMER run.

Outputs written to merger/outputs/stage2/:
  stage2_params.yaml
  scalars.txt
  ic_radial_profiles.png
  stability_profiles.png
  pressure_profiles.png
  density_slices.png

Usage:
  conda run -n jax-gpu python merger/stage2_gas_dm_halo.py [--n-steps N] [--quick]
"""

from __future__ import annotations

import os
import sys
import yaml
import time
import argparse
import imageio.v2 as imageio
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from astropy import units as us

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import diffhydro as dh
from diffhydro.equationmanager import EquationManager
from merger.physical_pm_force import PhysicalDMGravityForce, G_PHYS, _poisson_accel, _kick_gas
from merger.stage1_dm_halo import build_ic as build_dm_particle_ic, R200

# ─── output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(_HERE, "outputs", "stage2")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── physical constants / halo parameters ────────────────────────────────────
Z       = 0.295
G       = G_PHYS    # kpc^3 Msun^-1 Myr^-2
GAMMA   = 5.0 / 3.0
L_BOX   = 6144.0    # kpc  (matches GAMER run)
N_GRID  = 64        # cells per side

# Profile from gamer_single_cluster/gen_ics.py (same ICs as the GAMER run)
PROFILE_H5 = os.path.join(_HERE, "gamer_single_cluster", "profile1.h5")


# ─── argument parsing ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Gas+DM single-halo DiffHydro run")
    p.add_argument("--n-steps", type=int,   default=400,
                   help="Number of evolution steps (default 400, ~10 Gyr)")
    p.add_argument("--max-dt",  type=float, default=50.0,
                   help="Maximum timestep in Myr (default 50)")
    p.add_argument("--snapshot-every", type=int, default=40,
                   help="Snapshot cadence in steps (default 40)")
    p.add_argument("--dm-mode", type=str, default="particles", choices=["particles", "static"],
                   help="Use live DM particles or a fixed analytic DM density field")
    p.add_argument("--n-par", type=int, default=64**3,
                   help="Number of live DM particles when --dm-mode=particles")
    p.add_argument("--r-max", type=float, default=None,
                   help="DM particle truncation radius [kpc] when --dm-mode=particles (default 2.5*r200)")
    p.add_argument("--gravity-method", type=str, default="pm", choices=["pm", "direct"],
                   help="DM particle gravity method for live particles; gas still uses the grid field")
    p.add_argument("--softening-cells", type=float, default=0.15,
                   help="Plummer softening length for direct live-DM particle forces, in cell units")
    p.add_argument("--quick",   action="store_true",
                   help="Quick test: n_steps=20, snapshot_every=5")
    return p.parse_args()


# ─── IC builders ─────────────────────────────────────────────────────────────
def _interp_3d(r_prof, val_prof, n_grid, l_box):
    """Interpolate a 1D radial profile onto a 3D grid centred at box-centre."""
    dx  = l_box / n_grid
    x   = (np.arange(n_grid) + 0.5) * dx
    cen = l_box / 2.0
    xi, yi, zi = np.meshgrid(x - cen, x - cen, x - cen, indexing="ij")
    r3d = np.sqrt(xi**2 + yi**2 + zi**2)
    r3d_clamp = np.clip(r3d, r_prof[0], r_prof[-1])
    return np.interp(r3d_clamp, r_prof, val_prof).astype(np.float32)


def build_gas_state(profile_h5, n_grid, l_box):
    """3D conserved gas state [rho, 0, 0, 0, E] from HSE profile.

    Units: density Msun/kpc³, energy density Msun kpc⁻¹ Myr⁻².
    """
    with h5py.File(profile_h5, "r") as f:
        r_prof    = f["fields"]["radius"][:]
        rho_prof  = f["fields"]["density"][:]
        pres_prof = f["fields"]["pressure"][:]

    rho3d  = np.maximum(_interp_3d(r_prof, rho_prof,  n_grid, l_box), 1e-8)
    pres3d = np.maximum(_interp_3d(r_prof, pres_prof, n_grid, l_box), 1e-12)
    E3d    = pres3d / (GAMMA - 1.0)
    zeros  = np.zeros_like(rho3d)
    return np.stack([rho3d, zeros, zeros, zeros, E3d], axis=0)  # (5, N, N, N)


def build_dm_density_field(profile_h5, n_grid, l_box):
    """3D analytic DM density = total_density − gas_density from profile.

    Passing this as static_density_field to PhysicalDMGravityForce makes the
    Poisson source rho_DM_analytic + rho_gas ≈ rho_total — matching the total
    density against which cluster_generator integrated the HSE pressure.
    This sidesteps CIC softening (~Δx = 96 kpc) that would otherwise
    under-resolve gravity in the cluster core and cause the gas to explode.
    """
    with h5py.File(profile_h5, "r") as f:
        r_prof     = f["fields"]["radius"][:]
        rho_t_prof = f["fields"]["total_density"][:]
        rho_g_prof = f["fields"]["density"][:]

    rho_dm_prof = np.maximum(rho_t_prof - rho_g_prof, 0.0)
    return np.maximum(_interp_3d(r_prof, rho_dm_prof, n_grid, l_box), 0.0)


# ─── radial profile helper ───────────────────────────────────────────────────
def radial_profile_3d(field_3d, n_grid, l_box, nbins=80):
    """Spherical average of a 3D grid field.  Returns (r_kpc, mean_values)."""
    dx  = l_box / n_grid
    cen = n_grid / 2.0
    ix, iy, iz = np.indices((n_grid, n_grid, n_grid)) + 0.5
    r_kpc = np.sqrt((ix - cen)**2 + (iy - cen)**2 + (iz - cen)**2) * dx
    bins  = np.linspace(0, r_kpc.max(), nbins + 1)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    vals  = np.zeros(nbins)
    for i in range(nbins):
        mask = (r_kpc >= bins[i]) & (r_kpc < bins[i+1])
        if mask.sum() > 0:
            vals[i] = field_3d[mask].mean()
    return r_mid, vals


# ─── evolution ───────────────────────────────────────────────────────────────
def build_dm_particle_state(n_par, r_max, l_box):
    """Generate live DM particles and shift them into box coordinates."""
    pos0, vel0, m_par, ref = build_dm_particle_ic(n_par, r_max)
    center = np.array([l_box / 2.0] * 3, dtype=np.float32)
    pos_box = (pos0 + center[None, :]).astype(np.float32)
    vel0 = vel0.astype(np.float32)
    return {
        "pos": pos_box,
        "vel": vel0,
        "m_par": np.float32(m_par),
        "pos_centered": pos0.astype(np.float32),
        "ref": ref,
    }


class DirectParticleHybridGravityForce(PhysicalDMGravityForce):
    """Direct DM particle forces with PM gas coupling.

    DM particle accelerations are computed by direct summation with Plummer-like
    softening, while the gas kick continues to use the PM field sourced by the
    combined DM+gas density on the mesh.
    """

    def __init__(self, N_grid, L_box_kpc, *, softening_cells=0.5, **kwargs):
        super().__init__(N_grid, L_box_kpc, **kwargs)
        self.softening_cells = float(softening_cells)

    def _direct_particle_accel(self, pos_kpc, masses):
        delta = pos_kpc[:, None, :] - pos_kpc[None, :, :]
        delta = jnp.mod(delta + 0.5 * self.L, self.L) - 0.5 * self.L
        r2 = jnp.sum(delta * delta, axis=-1)
        eps2 = (self.softening_cells * self.dx) ** 2
        mask = 1.0 - jnp.eye(pos_kpc.shape[0], dtype=jnp.float32)
        inv_r3 = mask * jax.lax.rsqrt(jnp.maximum(r2 + eps2, 1.0e-20)) ** 3
        pair = -self.G * masses[None, :, None] * delta * inv_r3[..., None]
        return jnp.sum(pair, axis=1)

    def force(self, i_step, U, params, dt):
        dt = jnp.maximum(jnp.asarray(dt, dtype=jnp.float32), 0.0)
        dt_half = 0.5 * dt

        dm = params.get("dm", None)
        if dm is None:
            return U, params

        pos = jnp.asarray(dm["pos"], dtype=jnp.float32)
        vel = jnp.asarray(dm["vel"], dtype=jnp.float32)
        m_par = jnp.asarray(dm["m_par"], dtype=jnp.float32)
        if m_par.ndim == 0:
            masses = jnp.ones((pos.shape[0],), dtype=jnp.float32) * m_par
        else:
            masses = m_par

        rho_dm_old = self._deposit_density(pos, masses)
        rho_gas = jnp.asarray(U[0], dtype=jnp.float32) * self.gas_unit if self.include_gas else 0.0
        rho_src_old = rho_dm_old + rho_gas
        if self.static_density is not None:
            rho_src_old = rho_src_old + self.static_density

        a_par_old = self._direct_particle_accel(pos, masses)
        vel_half = vel + dt_half * a_par_old
        pos_new = jnp.mod(pos + dt * vel_half, self.L)

        rho_dm_new = self._deposit_density(pos_new, masses)
        rho_src_new = rho_dm_new + rho_gas
        if self.static_density is not None:
            rho_src_new = rho_src_new + self.static_density

        ax_new, ay_new, az_new = _poisson_accel(
            rho_src_new,
            self.kx_r, self.ky_r, self.kz_r, self.k2_r,
            self.G, self.subtract_mean, self.mesh_shape,
        )

        a_par_new = self._direct_particle_accel(pos_new, masses)
        vel_new = vel_half + dt_half * a_par_new

        rho_g = jnp.maximum(jnp.asarray(U[0], dtype=jnp.float32), self.eps)
        U_new = _kick_gas(U, ax_new, ay_new, az_new, rho_g, dt, self.eps)

        params_out = dict(params)
        dm_out = dict(dm)
        dm_out["pos"] = pos_new
        dm_out["vel"] = vel_new
        params_out["dm"] = dm_out
        return U_new, params_out


def run_evolution(U0, rho_dm_3d, n_grid, l_box, n_steps, snapshot_every, max_dt, *, dm_particles=None, gravity_method="pm", softening_cells=0.5):
    """Evolve gas in analytic DM potential with DiffHydro.

    If ``dm_particles`` is None, the DM density is fixed (static_density_field)
    and gas self-gravity is live. Otherwise, live DM particles are evolved and
    gas self-gravity is included in the same Poisson solve.

    Returns U_snaps, t_Myr, t_wall, dm_pos_snaps.
    """
    if dm_particles is not None and gravity_method == "direct":
        dm_label = "live DM particles (direct) + gas PM gravity"
    elif dm_particles is not None:
        dm_label = "live DM particles (PM) + live gas gravity"
    else:
        dm_label = "static DM + live gas gravity"
    print(f"\n[Stage 2] Evolving: N_grid={n_grid}, n_steps={n_steps}, "
          f"max_dt={max_dt} Myr  [{dm_label}]")

    eq = EquationManager()
    eq.gamma      = GAMMA
    eq.mesh_shape = [n_grid, n_grid, n_grid]
    eq.box_size   = (l_box, l_box, l_box)

    force_cls = DirectParticleHybridGravityForce if (dm_particles is not None and gravity_method == "direct") else PhysicalDMGravityForce
    force = force_cls(
        n_grid, l_box,
        G=G,
        subtract_mean=True,
        cfl_ff=0.3,
        include_gas_in_gravity=True,
        static_density_field=None if dm_particles is not None else rho_dm_3d,
        softening_cells=softening_cells,
    ) if force_cls is DirectParticleHybridGravityForce else force_cls(
        n_grid, l_box,
        G=G,
        subtract_mean=True,
        cfl_ff=0.3,
        include_gas_in_gravity=True,
        static_density_field=None if dm_particles is not None else rho_dm_3d,
    )

    ss     = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    flux   = dh.ConvectiveFlux(
        eq, solver, dh.PPM_CW(limiter="VANLEER"), positivity=True
    )
    flux.dx_o = l_box / n_grid

    if dm_particles is None:
        # Dummy zero-mass particle so params["dm"] is present.
        params_curr = {
            "dm": {
                "pos":   np.array([[l_box/2, l_box/2, l_box/2]], dtype=np.float32),
                "vel":   np.zeros((1, 3), dtype=np.float32),
                "m_par": np.float32(0.0),
            }
        }
    else:
        params_curr = {
            "dm": {
                "pos": np.asarray(dm_particles["pos"], dtype=np.float32),
                "vel": np.asarray(dm_particles["vel"], dtype=np.float32),
                "m_par": np.float32(dm_particles["m_par"]),
            }
        }

    U_curr  = U0.astype(np.float32)
    U_snaps = [U0.copy()]
    dm_pos_snaps = [] if dm_particles is None else [np.asarray(params_curr["dm"]["pos"], dtype=np.float32)]
    t_myr = [0.0]
    steps_done = 0
    sims = {}

    t_wall_start = time.time()
    while steps_done < n_steps:
        n_chunk = min(snapshot_every, n_steps - steps_done)
        if n_chunk not in sims:
            sims[n_chunk] = dh.hydro(
                n_super_step=n_chunk,
                max_dt=max_dt,
                fluxes=[flux],
                forces=[force],
                use_mol=True,
                pmesh_shape=(1, 1, 1),
                dx_o=flux.dx_o,
            )
        U_curr, params_curr, dt_hist = sims[n_chunk].evolve_with_callbacks(U_curr, params_curr)
        steps_done += n_chunk
        U_snaps.append(np.array(U_curr))
        if dm_particles is not None:
            dm_pos_snaps.append(np.asarray(params_curr["dm"]["pos"], dtype=np.float32))
        dt_chunk = float(np.sum(np.asarray(dt_hist, dtype=np.float64)))
        t_myr.append(t_myr[-1] + dt_chunk)
        print(f"  step {steps_done}/{n_steps}", flush=True)

    t_wall = time.time() - t_wall_start
    print(f"  Wall time: {t_wall:.1f} s")

    return U_snaps, np.asarray(t_myr, dtype=np.float64), t_wall, dm_pos_snaps


# ─── diagnostic plots ────────────────────────────────────────────────────────
def plot_ic_radial_profiles(U0, r_prof, rho_prof, pres_prof, n_grid, l_box):
    rho3d  = np.array(U0[0])
    pres3d = np.array(U0[4]) * (GAMMA - 1.0)
    r_mid, rho0  = radial_profile_3d(rho3d,  n_grid, l_box)
    r_mid, pres0 = radial_profile_3d(pres3d, n_grid, l_box)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 2 IC: Gas radial profiles vs cluster_generator target")

    ax = axes[0]
    m = rho0 > 0
    ax.loglog(r_mid[m], rho0[m],                        "C0-",  lw=2, label="DiffHydro IC")
    ax.loglog(r_mid[m], np.interp(r_mid, r_prof, rho_prof)[m],  "k--", lw=2, label="HSE target")
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("ρ (Msun/kpc³)")
    ax.set_title("Gas density"); ax.legend(); ax.grid(ls=":", alpha=0.5)

    ax = axes[1]
    m = pres0 > 0
    ax.loglog(r_mid[m], pres0[m],                         "C1-",  lw=2, label="DiffHydro IC")
    ax.loglog(r_mid[m], np.interp(r_mid, r_prof, pres_prof)[m],  "k--", lw=2, label="HSE target")
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("P (Msun kpc⁻¹ Myr⁻²)")
    ax.set_title("Thermal pressure"); ax.legend(); ax.grid(ls=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ic_radial_profiles.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out}")


def plot_stability(U_snaps, t_Myr, r_prof, rho_prof, pres_prof, n_grid, l_box):
    U0 = U_snaps[0];  UF = U_snaps[-1]
    t0 = t_Myr[0];   tF = t_Myr[-1]

    rho0_3d  = np.array(U0[0]);  rhoF_3d  = np.array(UF[0])
    pres0_3d = np.array(U0[4]) * (GAMMA - 1.0)
    presF_3d = np.array(UF[4]) * (GAMMA - 1.0)

    r_mid, rho0  = radial_profile_3d(rho0_3d,  n_grid, l_box)
    r_mid, rhoF  = radial_profile_3d(rhoF_3d,  n_grid, l_box)
    r_mid, pres0 = radial_profile_3d(pres0_3d, n_grid, l_box)
    r_mid, presF = radial_profile_3d(presF_3d, n_grid, l_box)

    rho_tgt  = np.interp(r_mid, r_prof, rho_prof)
    pres_tgt = np.interp(r_mid, r_prof, pres_prof)

    m0 = float(np.array(U0[0]).sum()) * (l_box / n_grid)**3
    dm_frac = [(float(np.array(U[0]).sum()) * (l_box / n_grid)**3 - m0) / m0 * 100.0
               for U in U_snaps]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Stage 2 DiffHydro: Gas+DM Single-Cluster Stability  "
        f"[static analytic DM + live gas gravity]\n"
        f"M₂₀₀=5×10¹⁴ M☉, z=0.295, {n_grid}³, "
        f"L={l_box:.0f} kpc, Δx={l_box/n_grid:.0f} kpc/cell",
        fontsize=11,
    )

    ax = axes[0, 0]
    m0m = rho0 > 0;  mFm = rhoF > 0;  mtm = rho_tgt > 0
    ax.loglog(r_mid[m0m], rho0[m0m], "C0-",  lw=2.5, label=f"t={t0:.0f} Myr (IC)")
    ax.loglog(r_mid[mFm], rhoF[mFm], "C1--", lw=2.5, label=f"t={tF:.0f} Myr (final)")
    ax.loglog(r_mid[mtm], rho_tgt[mtm], "k:", lw=2, label="HSE target")
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("ρ (Msun/kpc³)")
    ax.set_title("Gas Density Profile"); ax.legend(); ax.grid(ls=":", alpha=0.5)

    ax = axes[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.abs(rhoF - rho0) / np.maximum(rho0, 1e-30)
    ax.semilogx(r_mid, frac * 100.0, "C2-", lw=2)
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("|Δρ|/ρ₀ (%)")
    ax.set_title(f"Fractional Density Change (t={tF:.0f} Myr)")
    ax.set_ylim(0, None); ax.grid(ls=":", alpha=0.5)

    ax = axes[1, 0]
    ax.plot(t_Myr, dm_frac, "C3-", lw=2)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("t (Myr)"); ax.set_ylabel("ΔM_gas/M₀ (%)")
    ax.set_title("Gas Mass Conservation"); ax.grid(ls=":", alpha=0.5)

    ax = axes[1, 1]
    sl0 = rho0_3d[:, :, n_grid // 2]
    L   = l_box / 2
    vmin = max(float(sl0.min()), float(sl0.max()) * 1e-4)
    im = ax.imshow(sl0.T, origin="lower",
                   norm=LogNorm(vmin=vmin, vmax=float(sl0.max())),
                   cmap="plasma", extent=[-L, L, -L, L], aspect="equal")
    plt.colorbar(im, ax=ax, label="ρ (Msun/kpc³)")
    ax.set_xlabel("x (kpc)"); ax.set_ylabel("y (kpc)")
    ax.set_title(f"IC Density Slice (z-midplane, t={t0:.0f} Myr)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "stability_profiles.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out}")

    # Pressure figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Stage 2: Gas Pressure Radial Profiles")

    ax = axes2[0]
    m0p = pres0 > 0;  mFp = presF > 0;  mtp = pres_tgt > 0
    ax.loglog(r_mid[m0p], pres0[m0p], "C0-",  lw=2.5, label=f"t={t0:.0f} Myr (IC)")
    ax.loglog(r_mid[mFp], presF[mFp], "C1--", lw=2.5, label=f"t={tF:.0f} Myr (final)")
    ax.loglog(r_mid[mtp], pres_tgt[mtp], "k:", lw=2, label="HSE target")
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("P (Msun kpc⁻¹ Myr⁻²)")
    ax.set_title("Thermal Pressure Profile"); ax.legend(); ax.grid(ls=":", alpha=0.5)

    ax = axes2[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_p = np.abs(presF - pres0) / np.maximum(pres0, 1e-30)
    ax.semilogx(r_mid, frac_p * 100.0, "C4-", lw=2)
    ax.set_xlabel("r (kpc)"); ax.set_ylabel("|ΔP|/P₀ (%)")
    ax.set_title(f"Fractional Pressure Change (t={tF:.0f} Myr)")
    ax.set_ylim(0, None); ax.grid(ls=":", alpha=0.5)

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "pressure_profiles.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out2}")


def plot_density_slices(U_snaps, t_Myr, n_grid, l_box):
    U0 = U_snaps[0];  UF = U_snaps[-1]
    sl0 = np.array(U0[0])[:, :, n_grid // 2]
    slF = np.array(UF[0])[:, :, n_grid // 2]
    L   = l_box / 2.0
    ext = [-L, L, -L, L]
    vmin = max(float(sl0.min()), float(sl0.max()) * 1e-4)
    vmax = float(sl0.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, slc, tMyr, tag in [
        (axes[0], sl0, t_Myr[0],  "IC"),
        (axes[1], slF, t_Myr[-1], "Final"),
    ]:
        im = ax.imshow(slc.T, origin="lower",
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       cmap="plasma", extent=ext, aspect="equal")
        fig.colorbar(im, ax=ax, label="ρ (Msun/kpc³)")
        ax.set_xlabel("x (kpc)"); ax.set_ylabel("y (kpc)")
        ax.set_title(f"t = {tMyr:.0f} Myr  ({tag})")
    fig.suptitle("Stage 2 DiffHydro: Gas Density Slice (z-midplane)")
    out = os.path.join(OUT_DIR, "density_slices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out}")


def radial_profile_particles_box(pos_box, m_par, l_box, nbins=80):
    center = np.array([l_box / 2.0] * 3, dtype=np.float64)
    r = np.sqrt(np.sum((np.asarray(pos_box, dtype=np.float64) - center[None, :]) ** 2, axis=1))
    bins = np.linspace(0.0, 0.5 * np.sqrt(3.0) * l_box, nbins + 1)
    shell_mass, _ = np.histogram(r, bins=bins, weights=np.full(r.shape, float(m_par)))
    vol = 4.0 / 3.0 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    rho = shell_mass / np.maximum(vol, 1.0e-30)
    return r_mid, rho


def plot_dm_profile_evolution(dm_pos_snaps, t_myr, m_par, n_grid, l_box):
    if not dm_pos_snaps:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    pick = np.unique(np.linspace(0, len(dm_pos_snaps) - 1, min(5, len(dm_pos_snaps))).astype(int))
    for idx in pick:
        r_mid, rho = radial_profile_particles_box(dm_pos_snaps[idx], m_par, l_box, nbins=64)
        mask = rho > 0
        ax.loglog(r_mid[mask], rho[mask], lw=2, label=f"t={t_myr[idx]:.0f} Myr")
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel(r"$\rho_{DM}$ (Msun/kpc$^3$)")
    ax.set_title("Stage 2: DM density profile evolution")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    out = os.path.join(OUT_DIR, "dm_profile_evolution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


def make_dm_density_animation(dm_pos_snaps, t_myr, m_par, n_grid, l_box, out_dir):
    if not dm_pos_snaps or len(dm_pos_snaps) < 2:
        return

    force = PhysicalDMGravityForce(
        n_grid, l_box,
        G=G,
        subtract_mean=True,
        include_gas_in_gravity=False,
    )
    frames = []
    rho_slices = []
    for pos in dm_pos_snaps:
        rho = np.asarray(force._deposit_density(jnp.asarray(pos, dtype=jnp.float32), jnp.asarray(m_par, dtype=jnp.float32)))
        rho_slices.append(rho[:, :, n_grid // 2])

    log_slices = [np.log10(np.maximum(sl, 1.0e-30)) for sl in rho_slices]
    vmin = 1.0#min(float(np.min(sl)) for sl in log_slices)
    vmax = max(float(np.max(sl)) for sl in log_slices)
    L = l_box / 2.0
    extent = [-L, L, -L, L]

    for t_now, rho_sl, log_sl in zip(t_myr, rho_slices, log_slices):
        frac = (rho_sl - rho_slices[0]) / np.maximum(rho_slices[0], 1.0e-30)
        lim = max(0.05, float(np.nanpercentile(np.abs(frac), 99.0)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = axes[0].imshow(log_sl.T, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        axes[0].set_title(rf"$\log_{{10}}\rho_{{DM}}$ at $t={t_now:.0f}$ Myr")
        axes[0].set_xlabel("x [kpc]")
        axes[0].set_ylabel("y [kpc]")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(frac.T, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal")
        axes[1].set_title("Fractional DM density change")
        axes[1].set_xlabel("x [kpc]")
        axes[1].set_ylabel("y [kpc]")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    out = os.path.join(out_dir, "dm_density_evolution.gif")
    imageio.mimsave(out, frames, duration=0.6, loop=0)
    print(f"  Saved {out}")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.quick:
        args.n_steps        = 20
        args.snapshot_every = 5
        if args.dm_mode == "particles":
            args.n_par = min(int(args.n_par), 4096)

    if not os.path.exists(PROFILE_H5):
        raise FileNotFoundError(
            f"{PROFILE_H5} not found.\n"
            "Run:  conda run -n gamer python merger/gamer_single_cluster/gen_ics.py"
        )

    # Load target 1D profiles
    with h5py.File(PROFILE_H5, "r") as f:
        r_prof    = f["fields"]["radius"][:]
        rho_prof  = f["fields"]["density"][:]
        pres_prof = f["fields"]["pressure"][:]

    # Build 3D gas IC and analytic DM density field
    print("[Stage 2] Building gas IC and DM source...")
    U0      = build_gas_state(PROFILE_H5, N_GRID, L_BOX)
    rho_dm  = build_dm_density_field(PROFILE_H5, N_GRID, L_BOX)
    dm_particles = None
    if args.dm_mode == "particles":
        dm_r_max = float(args.r_max) if args.r_max is not None else 2.5 * R200
        dm_particles = build_dm_particle_state(int(args.n_par), dm_r_max, L_BOX)

    rho_peak = float(np.max(U0[0]))
    m_gas    = float(np.sum(U0[0])) * (L_BOX / N_GRID)**3
    dm_peak  = float(np.max(rho_dm))
    print(f"  Gas: peak_rho={rho_peak:.3e} Msun/kpc³, total_mass={m_gas:.3e} Msun")
    if dm_particles is None:
        print(f"  DM static field: peak_rho={dm_peak:.3e} Msun/kpc³")
    else:
        print(f"  DM particles: N={int(args.n_par):,}, m_par={float(dm_particles['m_par']):.3e} Msun")

    # IC verification plot
    print("[Stage 2] Plotting IC radial profiles...")
    plot_ic_radial_profiles(U0, r_prof, rho_prof, pres_prof, N_GRID, L_BOX)

    # Evolve
    U_snaps, t_Myr, t_wall, dm_pos_snaps = run_evolution(
        U0, rho_dm, N_GRID, L_BOX,
        args.n_steps, args.snapshot_every, args.max_dt,
        dm_particles=dm_particles,
        gravity_method=args.gravity_method,
        softening_cells=args.softening_cells,
    )
    print(f"[Stage 2] Done. Wall time={t_wall:.1f} s, t_est={t_Myr[-1]:.0f} Myr")

    # Diagnostic plots
    print("[Stage 2] Plotting diagnostics...")
    plot_stability(U_snaps, t_Myr, r_prof, rho_prof, pres_prof, N_GRID, L_BOX)
    plot_density_slices(U_snaps, t_Myr, N_GRID, L_BOX)
    if dm_particles is not None:
        plot_dm_profile_evolution(dm_pos_snaps, t_Myr, dm_particles["m_par"], N_GRID, L_BOX)
        make_dm_density_animation(dm_pos_snaps, t_Myr, dm_particles["m_par"], N_GRID, L_BOX, OUT_DIR)

    # Scalars
    mF = float(np.sum(U_snaps[-1][0])) * (L_BOX / N_GRID)**3
    scalars = {
        "n_grid":              N_GRID,
        "l_box_kpc":           L_BOX,
        "dx_kpc":              L_BOX / N_GRID,
        "n_steps":             args.n_steps,
        "max_dt_Myr":          args.max_dt,
        "t_final_Myr_est":     float(t_Myr[-1]),
        "t_wall_s":            t_wall,
        "m_gas_initial_Msun":  m_gas,
        "m_gas_final_Msun":    mF,
        "dm_mass_frac_err":    (mF - m_gas) / m_gas,
        "rho_peak_initial":    rho_peak,
        "m_par_Msun":          float(dm_particles["m_par"]) if dm_particles is not None else 0.0,
    }
    scal_path = os.path.join(OUT_DIR, "scalars.txt")
    with open(scal_path, "w") as fh:
        for k, v in scalars.items():
            fh.write(f"{k} = {v}\n")
    print(f"  Saved {scal_path}")

    params = {
        "stage":             2,
        "z":                 Z,
        "L_box_kpc":         L_BOX,
        "N_grid":            N_GRID,
        "n_steps":           args.n_steps,
        "max_dt_Myr":        args.max_dt,
        "dm_source":         "live particles from cluster_generator" if dm_particles is not None else "static analytic field from profile1.h5",
        "gas_in_gravity":    True,
        "profile":           PROFILE_H5,
        "n_par":             int(args.n_par) if dm_particles is not None else 0,
        "r_max_kpc":         float(dm_r_max) if dm_particles is not None else None,
        "gravity_method":    str(args.gravity_method),
        "softening_cells":   float(args.softening_cells),
    }
    yaml_path = os.path.join(OUT_DIR, "stage2_params.yaml")
    with open(yaml_path, "w") as fh:
        yaml.dump(params, fh, default_flow_style=False)
    print(f"  Saved {yaml_path}")

    print(f"\n[Stage 2] Complete.  Outputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
