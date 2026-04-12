"""
stage1_dm_halo.py
=================
Stage 1: Single-halo, DM-only, non-cosmological run.

Goal: Validate that the NFW particle IC is correct and that the halo
remains in approximate virial equilibrium over a few dynamical times
under the non-cosmological PM gravity implemented in physical_pm_force.py.

What this script does:
  1. Build NFW halo IC via cluster_generator (M200=5e14 Msun, z=0.295)
  2. Validate the IC analytically (no evolution needed):
       - Total particle mass vs analytic M(<r_max)
       - Enclosed mass profile vs NFW analytic
       - Centre-of-mass offset
       - Net momentum (should be ~0)
  3. Short DiffHydro evolution (N_par=64^3, N_grid=64):
       - Pure DM gravity (no interesting gas physics)
       - Save density slice + halo profile snapshots every 10 steps
  4. Post-evolution diagnostics:
       - Density profile drift before vs after
       - Virial ratio 2K/|U|
       - COM drift

Outputs written to merger/outputs/stage1/:
  stage1_params.yaml
  scalars.txt
  ic_enclosed_mass.png
  ic_radial_number_density.png
  ic_density_slice.png
  ic_acceleration_vs_radius.png
  ic_phase_space.png
  tracked_trajectories.png
  stability_profile.png

Usage:
  conda run -n jax-gpu python merger/stage1_dm_halo.py [--n-par N] [--n-steps N] [--quick]
  conda run -n jax-gpu python merger/stage1_dm_halo.py --orbit-demo
"""

from __future__ import annotations

import os
import sys
import yaml
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from merger.physical_pm_force import (
    PhysicalDMGravityForce,
    nfw_density, nfw_enclosed_mass, nfw_acceleration, nfw_circular_velocity,
    G_PHYS,
)
from merger.halo_reference import HaloReference
import cluster_generator as cg

# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------
OUT_DIR = os.path.join(_HERE, "outputs", "stage1")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Physical constants / halo parameters
# -------------------------------------------------------------------------
Z       = 0.295
COS     = FlatLambdaCDM(H0=70.0, Om0=0.3)
RHOC    = COS.critical_density(Z).to(us.M_sun / us.kpc**3).value   # Msun/kpc^3

M200    = 5.0e14    # Msun
CONC    = 3.5
R200    = cg.find_overdensity_radius(M200, 200.0, z=Z)              # kpc
A_SCALE = R200 / CONC                                               # NFW scale radius
RHOS    = 200.0 * RHOC * CONC**3 / ((np.log(1.0 + CONC) - CONC / (1.0 + CONC)) * 3)

G       = G_PHYS    # kpc^3 Msun^-1 Myr^-2
GAMMA   = 5.0 / 3.0


def build_gamer_matched_cluster_model(
    *,
    m200: float = M200,
    conc: float = CONC,
    z: float = Z,
):
    """Build the same cluster_generator halo model used by the GAMER IC scripts."""
    r200 = cg.find_overdensity_radius(m200, 200.0, z=z)
    a = r200 / conc
    m_tot = cg.snfw_total_mass(m200, r200, a)
    mt = cg.snfw_mass_profile(m_tot, a)
    r500, m500 = cg.find_radius_mass(mt, z=z, delta=500.0)
    r2500, _ = cg.find_radius_mass(mt, z=z, delta=2500.0)
    rhot = cg.snfw_density_profile(m_tot, a)
    f_g = cg.f_gas(m500)
    rhog = cg.vikhlinin_density_profile(1.0, 0.2 * r2500, 0.67 * r200, 1.0, 0.67, 3.0)
    rhog = cg.rescale_profile_by_mass(rhog, f_g * m500, r500)
    model = cg.ClusterModel.from_dens_and_tden(0.1, 20000.0, rhog, rhot)
    return model, r200


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: DM-only single-halo test")
    p.add_argument("--n-par",   type=int,   default=64**3,
                   help="Number of DM particles (default 64^3=262144)")
    p.add_argument("--n-grid",  type=int,   default=64,
                   help="Grid resolution (default 64)")
    p.add_argument("--n-steps", type=int,   default=200,
                   help="Number of evolution steps (default 200)")
    p.add_argument("--max-dt",  type=float, default=5.0,
                   help="Maximum timestep in Myr (default 5)")
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="Snapshot cadence in steps (default n_steps//10)")
    p.add_argument("--r-max",   type=float, default=None,
                   help="Particle truncation radius kpc (default 2.5*r200)")
    p.add_argument("--l-box",   type=float, default=None,
                   help="Box side length kpc (default 4*r200)")
    p.add_argument("--quick",   action="store_true",
                   help="Quick test: N_par=4096, n_steps=20 (for CI / debugging)")
    p.add_argument("--orbit-demo", action="store_true",
                   help="Longer run aimed at showing near-orbital particle tracks")
    return p.parse_args()


# -------------------------------------------------------------------------
# IC generation
# -------------------------------------------------------------------------

def build_ic(n_par, r_max, seed=42, *, ic_model: str = "gamer"):
    """Generate DM particles using cluster_generator.

    Returns
    -------
    pos : (n_par, 3) float64, positions centred at origin [kpc]
    vel : (n_par, 3) float64, velocities [kpc/Myr]
    m_par : float, mass per particle [Msun]
    hse_model : cluster_generator.HydrostaticEquilibrium
    """
    if ic_model == "gamer":
        model, r200 = build_gamer_matched_cluster_model(m200=M200, conc=CONC, z=Z)
        ref = HaloReference(z=Z, m200_msun=M200, conc=CONC, r_grid_max=2.0 * r_max, star_fraction=0.0)
    elif ic_model == "legacy":
        ref = HaloReference(z=Z, m200_msun=M200, conc=CONC, r_grid_max=2.0 * r_max)
        model = ref.hse
        r200 = ref.r200
    else:
        raise ValueError(f"Unsupported ic_model={ic_model!r}")

    np.random.seed(seed)
    particles = model.generate_dm_particles(n_par, r_max=r_max)
    pos   = np.array(particles[("dm", "particle_position")])  # kpc, centered at 0
    vel   = np.array(particles[("dm", "particle_velocity")])  # kpc/Myr
    m_par = float(np.array(particles[("dm", "particle_mass")])[0])

    return pos, vel, m_par, ref


# -------------------------------------------------------------------------
# IC validation metrics
# -------------------------------------------------------------------------

def validate_ic(pos, vel, m_par, r_max, ref):
    """Compute IC validation scalar metrics.

    Returns a dict of named scalar diagnostics.
    """
    n_par  = len(pos)
    r_par  = np.sqrt(np.sum(pos**2, axis=1))       # radii from halo center
    M_total_par = n_par * m_par

    # Analytic quantities
    M200_total = nfw_enclosed_mass(R200, RHOS, A_SCALE)
    M_rmax_total = nfw_enclosed_mass(r_max, RHOS, A_SCALE)
    M200_dm = float(ref.mass_dm_at([R200])[0])
    M_rmax_dm = float(ref.mass_dm_at([r_max])[0])

    # Total mass
    frac_mass_err_dm = (M_total_par - M_rmax_dm) / M_rmax_dm

    # COM
    com = np.mean(pos, axis=0)
    com_offset = np.linalg.norm(com)                # should be ~0
    com_offset_over_r200 = com_offset / R200

    # Net momentum
    p_tot = np.sum(vel * m_par, axis=0)             # Msun kpc/Myr
    v_com = p_tot / M_total_par                     # kpc/Myr
    v_com_mag = np.linalg.norm(v_com)

    # Particles outside r_max
    n_outside = int(np.sum(r_par > r_max))

    # Kinetic energy
    v2    = np.sum(vel**2, axis=1)
    K_tot = 0.5 * m_par * np.sum(v2)                # Msun kpc^2/Myr^2

    diags = {
        "n_par": n_par,
        "m_par_Msun": m_par,
        "M_total_Msun": M_total_par,
        "M200_total_analytic_Msun": M200_total,
        "M_rmax_total_analytic_Msun": M_rmax_total,
        "M200_dm_target_Msun": M200_dm,
        "M_rmax_dm_target_Msun": M_rmax_dm,
        "frac_mass_error_dm": frac_mass_err_dm,
        "COM_offset_kpc": float(com_offset),
        "COM_offset_over_r200": float(com_offset_over_r200),
        "v_COM_kpc_per_Myr": float(v_com_mag),
        "v_COM_xyz": v_com.tolist(),
        "n_outside_rmax": n_outside,
        "K_total_Msun_kpc2_Myr2": float(K_tot),
    }

    print("\n[Stage 1] IC validation:")
    print(f"  N_par         = {n_par}")
    print(f"  M_total_par   = {M_total_par:.4e} Msun")
    print(f"  DM M(<r_max)  = {M_rmax_dm:.4e} Msun")
    print(f"  total M(<r_max)= {M_rmax_total:.4e} Msun")
    print(f"  DM mass_err   = {frac_mass_err_dm*100:.2f}%")
    print(f"  COM offset    = {com_offset:.2f} kpc  ({com_offset_over_r200:.4f} r200)")
    print(f"  |v_COM|       = {v_com_mag:.4e} kpc/Myr")
    print(f"  n_outside     = {n_outside}")
    return diags


# -------------------------------------------------------------------------
# Radial profile helpers
# -------------------------------------------------------------------------

def radial_profile_particles(pos, m_par, r_bins):
    """Compute particle density profile (Msun/kpc^3) in radial bins."""
    r_par = np.sqrt(np.sum(pos**2, axis=1))
    counts, _ = np.histogram(r_par, bins=r_bins)
    vol_shells = 4.0 / 3.0 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    rho_par = counts * m_par / vol_shells
    return rho_par, counts


def enclosed_mass_particles(pos, m_par, r_eval):
    """Enclosed mass from particles at each radius in r_eval."""
    r_par = np.sqrt(np.sum(pos**2, axis=1))
    return np.array([m_par * np.sum(r_par < r) for r in r_eval])


def circular_period(r):
    """Local circular orbital period in Myr for the analytic total NFW profile."""
    r = np.asarray(r, dtype=np.float64)
    vc = nfw_circular_velocity(r, RHOS, A_SCALE, G=G)
    return 2.0 * np.pi * r / np.maximum(vc, 1.0e-12)


# -------------------------------------------------------------------------
# IC diagnostic plots
# -------------------------------------------------------------------------

def plot_ic_diagnostics(pos, vel, m_par, force, n_par, r_max, l_box, ref):
    """Produce all Stage 1 IC diagnostic figures."""
    r_par = np.sqrt(np.sum(pos**2, axis=1))
    vr    = np.sum(vel * (pos / (r_par[:, None] + 1e-20)), axis=1)

    r_bins = np.logspace(np.log10(0.02 * R200), np.log10(1.5 * r_max), 55)
    r_cen  = 0.5 * (r_bins[:-1] + r_bins[1:])
    M_enc_par = enclosed_mass_particles(pos, m_par, r_cen)
    M_enc_dm = ref.mass_dm_at(r_cen)
    M_enc_total = nfw_enclosed_mass(r_cen, RHOS, A_SCALE)
    rho_par, counts = radial_profile_particles(pos, m_par, r_bins)

    # Shift positions to box for deposition
    dx = l_box / force.N
    CENTER = np.array([l_box / 2.0] * 3)
    pos_box = pos + CENTER[None, :]

    # ---- 1. Enclosed mass ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(r_cen / R200, M_enc_par, "o-", ms=3, lw=1.5, label="Particles")
    ax.loglog(r_cen / R200, M_enc_dm, "k--", lw=2, label="DM target")
    ax.loglog(r_cen / R200, M_enc_total, color="gray", ls=":", lw=1.5, label="Total NFW")
    ax.axvline(1.0, color="gray", ls=":", lw=1, label="r200")
    ax.axvline(r_max / R200, color="orange", ls="--", lw=1, label=f"r_max={r_max:.0f} kpc")
    ax.set_xlabel("r / r200"); ax.set_ylabel("M(<r) [Msun]")
    ax.set_title(f"Stage 1: Enclosed mass  (N_par={n_par:,})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_enclosed_mass.png"), dpi=120)
    plt.close()

    # ---- 2. Radial number density ----------------------------------------
    n_density = counts / (4.0/3.0 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3))
    rho_dm = ref.rho_dm_at(r_cen)
    rho_nfw_par = rho_dm / m_par   # convert to par/kpc^3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(r_cen, n_density + 1e-30, "o-", ms=3, lw=1.5, label="Particles n(r)")
    ax.loglog(r_cen, rho_nfw_par, "k--", lw=2, label="DM target / m_par")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel("n [par/kpc³]")
    ax.set_title("Stage 1: Particle radial number density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_radial_number_density.png"), dpi=120)
    plt.close()

    # ---- 3. Density slice (DM deposited on grid) -------------------------
    ax_field, ay_field, az_field = force.compute_accel_field(
        jnp.array(pos_box, dtype=jnp.float32), m_par
    )
    rho_grid = np.array(force._deposit_density(
        jnp.array(pos_box, dtype=jnp.float32), m_par
    ))

    N = force.N
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage 1: DM density and acceleration (z-mid slice)")

    im = axes[0].imshow(
        np.log10(rho_grid[:, :, N//2] + 1e-30),
        origin="lower", cmap="hot",
        extent=[0, l_box, 0, l_box],
    )
    axes[0].set_title(r"$\log_{10}(\rho_{DM})$ [Msun/kpc³]")
    axes[0].set_xlabel("x [kpc]"); axes[0].set_ylabel("y [kpc]")
    plt.colorbar(im, ax=axes[0])

    amag = np.sqrt(
        np.array(ax_field)**2 +
        np.array(ay_field)**2 +
        np.array(az_field)**2
    )
    im2 = axes[1].imshow(
        np.log10(amag[:, :, N//2] + 1e-30),
        origin="lower", cmap="viridis",
        extent=[0, l_box, 0, l_box],
    )
    axes[1].set_title(r"$\log_{10}(|\mathbf{a}|)$ [kpc/Myr²]")
    axes[1].set_xlabel("x [kpc]"); axes[1].set_ylabel("y [kpc]")
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_density_slice.png"), dpi=120)
    plt.close()

    # ---- 4. Acceleration magnitude vs radius ----------------------------
    pos_grid = np.array(force._pos_to_grid(jnp.array(pos_box, dtype=jnp.float32)))
    # sample acceleration at particle positions
    a_par = np.array(force._sample_accel_at_par(
        jnp.array(pos_grid, dtype=jnp.float32),
        ax_field, ay_field, az_field
    ))
    amag_par = np.linalg.norm(a_par, axis=1)

    r_anal = np.logspace(np.log10(0.02 * R200), np.log10(1.5 * r_max), 100)
    a_nfw = nfw_acceleration(r_anal, RHOS, A_SCALE, G=G)

    # Bin sampled acceleration by radius
    a_bins = np.logspace(np.log10(0.02 * R200), np.log10(1.5 * r_max), 40)
    a_cen  = 0.5 * (a_bins[:-1] + a_bins[1:])
    amag_binned, _ = np.histogram(r_par, bins=a_bins, weights=amag_par)
    cnt_bin, _     = np.histogram(r_par, bins=a_bins)
    mask = cnt_bin > 0
    amag_mean = amag_binned[mask] / cnt_bin[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r_par[::4], amag_par[::4], s=0.5, alpha=0.2, c="steelblue",
               label="Particles (sampled, 1 in 4)")
    ax.loglog(a_cen[mask], amag_mean, "ro-", ms=4, lw=1.5, label="Binned mean PM")
    ax.loglog(r_anal, a_nfw, "k--", lw=2, label="NFW analytic")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$|a|$ [kpc/Myr²]")
    ax.set_title("Stage 1: Acceleration magnitude vs radius")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_acceleration_vs_radius.png"), dpi=120)
    plt.close()

    # ---- 5. Phase-space plot: radius vs radial velocity -----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    # Subsample for plotting
    step = max(1, len(r_par) // 5000)
    ax.scatter(r_par[::step] / R200, vr[::step], s=0.5, alpha=0.3, c="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(1.0, color="red", lw=1, ls="--", label="r200")

    # Circular velocity for reference
    r_vc = np.linspace(0.05 * R200, 1.5 * r_max, 100)
    vc   = nfw_circular_velocity(r_vc, RHOS, A_SCALE, G=G)
    ax.plot(r_vc / R200,  vc, "r-", lw=1.5, alpha=0.7, label="±v_c(r)")
    ax.plot(r_vc / R200, -vc, "r-", lw=1.5, alpha=0.7)

    ax.set_xlabel("r / r200"); ax.set_ylabel("vr [kpc/Myr]")
    ax.set_title("Stage 1: Phase-space  r vs v_r")
    ax.legend()
    ax.set_xlim(0, r_max / R200 * 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_phase_space.png"), dpi=120)
    plt.close()

    print(f"[Stage 1] IC diagnostic plots written to {OUT_DIR}/")


# -------------------------------------------------------------------------
# Short DiffHydro evolution
# -------------------------------------------------------------------------

def build_gas_state(n_grid, l_box, rho_bg=None, cs_bg=0.1):
    """Minimal background gas state (uniform) for DM-only Stage 1 run."""
    if rho_bg is None:
        rho_bg = 1.0e-6 * RHOC   # Msun/kpc^3

    p_bg   = rho_bg * cs_bg**2 / GAMMA
    e_int  = p_bg / ((GAMMA - 1.0) * rho_bg)
    E_bg   = rho_bg * e_int

    rho = np.full((n_grid, n_grid, n_grid), rho_bg, dtype=np.float32)
    mx  = np.zeros_like(rho)
    my  = np.zeros_like(rho)
    mz  = np.zeros_like(rho)
    E   = np.full_like(rho, E_bg)
    return np.stack([rho, mx, my, mz, E], axis=0)  # (5, N, N, N)


def run_evolution(pos0, vel0, m_par, n_grid, l_box, n_steps, snapshot_every=20, max_dt=5.0):
    """Run DiffHydro with PhysicalDMGravityForce.

    Runs all steps in a single compiled call (n_super_step=n_steps) to avoid
    per-call JIT recompilation overhead.  For snapshots, runs in chunks of
    `snapshot_every` steps so intermediate states can be extracted.

    Returns
    -------
    pos_snapshots : list of (N_par, 3) arrays at snapshot times
    vel_snapshots : list of (N_par, 3) arrays
    times_Myr : list of floats
    """
    print(f"\n[Stage 1] Running evolution: N_grid={n_grid}, n_steps={n_steps}")

    eq = EquationManager()
    eq.gamma      = GAMMA
    eq.mesh_shape = [n_grid, n_grid, n_grid]
    eq.box_size   = (l_box, l_box, l_box)

    force = PhysicalDMGravityForce(
        n_grid, l_box,
        G=G,
        subtract_mean=True,
        cfl_ff=0.3,
        include_gas_in_gravity=False,
    )

    ss     = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    flux   = dh.ConvectiveFlux(
        eq, solver, dh.MUSCL3(limiter="VANLEER"), positivity=True
    )

    CENTER  = np.array([l_box / 2.0] * 3)
    pos_box = (pos0 + CENTER[None, :]).astype(np.float32)
    vel0f   = vel0.astype(np.float32)

    U0 = build_gas_state(n_grid, l_box).astype(np.float32)
    params0 = {
        "dm": {
            "pos":   pos_box,
            "vel":   vel0f,
            "m_par": np.float32(m_par),
        }
    }

    print(f"  JIT-compiling and running in chunks of {snapshot_every} step(s)...")
    t_wall_start = time.time()
    U_curr = U0
    params_curr = params0
    pos_snaps = [pos0.copy()]
    vel_snaps = [vel0.copy()]
    steps_done = 0

    sims = {}
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
            )
        U_curr, params_curr = sims[n_chunk].evolve(U_curr, params_curr)
        steps_done += n_chunk
        pos_snaps.append(np.array(params_curr["dm"]["pos"]) - CENTER[None, :])
        vel_snaps.append(np.array(params_curr["dm"]["vel"]))

    t_wall = time.time() - t_wall_start
    print(f"  Wall time (compile + run): {t_wall:.1f} s")

    # Estimate t_final from free-fall time and n_steps (evolve doesn't return dt)
    # Use the mean free-fall dt as a proxy
    rho_peak = float(np.max(build_gas_state(n_grid, l_box)[0]))
    from merger.physical_pm_force import G_PHYS
    dt_est = 0.3 * np.sqrt(3.0 * np.pi / (32.0 * G_PHYS * (
        m_par / (l_box / n_grid)**3 * 1.0  # rough peak density estimate
    ) + 1e-20))
    t_sim_est = n_steps * float(np.minimum(max_dt, dt_est))
    print(f"  Estimated final simulation time: ~{t_sim_est:.1f} Myr (n_steps × max_dt approx)")

    times = list(np.linspace(0.0, t_sim_est, len(pos_snaps)))
    return pos_snaps, vel_snaps, times


def select_tracked_particles(pos0, vel0, n_track=6):
    """Pick particles likely to show clear projected orbits in the xy plane."""
    r0 = np.sqrt(np.sum(pos0**2, axis=1))
    v0 = np.sqrt(np.sum(vel0**2, axis=1))
    vr = np.sum(pos0 * vel0, axis=1) / np.maximum(r0, 1.0e-20)
    vt = np.sqrt(np.maximum(v0**2 - vr**2, 0.0))
    vc = nfw_circular_velocity(np.maximum(r0, 1.0e-3), RHOS, A_SCALE, G=G)

    zfrac = np.abs(pos0[:, 2]) / np.maximum(r0, 1.0e-20)
    vzfrac = np.abs(vel0[:, 2]) / np.maximum(v0, 1.0e-20)
    circ = vt / np.maximum(vc, 1.0e-20)
    radiality = np.abs(vr) / np.maximum(v0, 1.0e-20)

    mask = (
        (r0 > 0.05 * R200)
        & (r0 < 0.16 * R200)
        & (zfrac < 0.20)
        & (vzfrac < 0.25)
        & (circ > 0.6)
        & (circ < 1.5)
        & (radiality < 0.45)
    )

    t_circ = circular_period(np.maximum(r0, 1.0))
    score = (
        np.abs(circ - 1.0)
        + 0.6 * radiality
        + 0.4 * zfrac
        + 0.3 * vzfrac
        + 0.6 * (r0 / R200)
    )
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) < n_track:
        candidate_idx = np.where((r0 > 0.04 * R200) & (r0 < 0.22 * R200))[0]
    if len(candidate_idx) < n_track:
        candidate_idx = np.arange(len(r0))
    order = candidate_idx[np.lexsort((score[candidate_idx], t_circ[candidate_idx]))]

    chosen = []
    for idx in order:
        if len(chosen) >= n_track:
            break
        if all(abs(r0[idx] - r0[j]) > 0.015 * R200 for j in chosen):
            chosen.append(int(idx))
    while len(chosen) < n_track:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen) >= n_track:
                    break
    return np.array(chosen[:n_track], dtype=int)


def plot_particle_trajectories(pos_snaps, vel_snaps, times, l_box, track_ids):
    """Plot late-time particle positions with a few tracked trajectories."""
    final_pos = pos_snaps[-1]
    step = max(1, len(final_pos) // 15000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax.scatter(
        final_pos[::step, 0],
        final_pos[::step, 1],
        s=0.4,
        alpha=0.15,
        color="0.6",
        label="Late-time particles",
    )

    cmap = plt.get_cmap("tab10")
    displacements = []
    orbit_fractions = []
    circ_periods = []
    for i, pid in enumerate(track_ids):
        traj = np.array([snap[pid] for snap in pos_snaps])
        vel_traj = np.array([snap[pid] for snap in vel_snaps])
        color = cmap(i % 10)
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.8, color=color, alpha=0.9)
        ax.plot(traj[0, 0], traj[0, 1], "o", ms=4, color=color)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", ms=6, mew=1.5, color=color)
        displacements.append(float(np.linalg.norm(traj[-1] - traj[0])))

        r0 = float(np.linalg.norm(traj[0]))
        circ_p = float(circular_period(max(r0, 1.0)))
        circ_periods.append(circ_p)
        theta = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
        orbit_fractions.append(float(np.abs(theta[-1] - theta[0]) / (2.0 * np.pi)))

    ax.set_xlim(-0.12 * l_box, 0.12 * l_box)
    ax.set_ylim(-0.12 * l_box, 0.12 * l_box)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title("Stage 1: Tracked particle trajectories (xy)")
    ax.add_patch(plt.Circle((0.0, 0.0), R200, color="k", fill=False, ls="--", lw=1.0))
    ax.text(
        0.02,
        0.98,
        f"snapshots={len(pos_snaps)}\n"
        f"t_end~{times[-1]:.0f} Myr\n"
        f"median |Δx|={np.median(displacements):.1f} kpc\n"
        f"median projected orbit frac={np.median(orbit_fractions):.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_aspect("equal")

    ax = axes[1]
    for i, pid in enumerate(track_ids):
        traj = np.array([snap[pid] for snap in pos_snaps])
        theta = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
        theta_cycles = (theta - theta[0]) / (2.0 * np.pi)
        ax.plot(times, theta_cycles, "-", lw=1.8, color=cmap(i % 10), label=f"id={pid}")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="1 projected orbit")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$\Delta \theta_{xy}/2\pi$")
    ax.set_title("Projected orbital phase accumulation")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tracked_trajectories.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[Stage 1] Trajectory figure: {out}")

    return {
        "tracked_particle_ids": track_ids.tolist(),
        "tracked_median_displacement_kpc": float(np.median(displacements)),
        "tracked_max_displacement_kpc": float(np.max(displacements)),
        "tracked_median_projected_orbit_fraction": float(np.median(orbit_fractions)),
        "tracked_max_projected_orbit_fraction": float(np.max(orbit_fractions)),
        "tracked_median_circular_period_Myr": float(np.median(circ_periods)),
    }


# -------------------------------------------------------------------------
# Post-evolution diagnostics
# -------------------------------------------------------------------------

def plot_stability(pos_snaps, vel_snaps, times, m_par, r_max, ref):
    """Compare density profile before and after evolution."""
    r_bins = np.logspace(np.log10(0.05 * R200), np.log10(1.2 * r_max), 50)
    r_cen  = 0.5 * (r_bins[:-1] + r_bins[1:])
    rho_dm = ref.rho_dm_at(r_cen)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Stage 1: Stability check — profile before vs after evolution")

    # Density profile at each snapshot
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, (pos, t) in enumerate(zip(pos_snaps, times)):
        r_par = np.sqrt(np.sum(pos**2, axis=1))
        rho_par, _ = radial_profile_particles(pos, m_par, r_bins)
        color = cmap(i / max(len(pos_snaps) - 1, 1))
        lw = 2.5 if (i == 0 or i == len(pos_snaps) - 1) else 0.8
        ax.loglog(r_cen, rho_par + 1e-30, color=color, lw=lw,
                  label=f"t={t:.0f} Myr" if (i == 0 or i == len(pos_snaps)-1) else "")
    ax.loglog(r_cen, rho_dm, "k--", lw=2, label="DM target")
    ax.set_ylim(10, 1E8)
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho$ [Msun/kpc³]")
    ax.set_title("Density profile evolution"); ax.legend(fontsize=8)

    # COM drift over time
    ax = axes[1]
    com_offsets = [np.linalg.norm(np.mean(pos, axis=0)) for pos in pos_snaps]
    ax.plot(times, com_offsets, "o-", ms=4)
    ax.axhline(0.01 * R200, color="gray", ls="--", label="0.01 r200")
    ax.set_xlabel("t [Myr]"); ax.set_ylabel("COM offset [kpc]")
    ax.set_title("Centre-of-mass drift"); ax.legend()

    # Kinetic energy over time
    ax = axes[2]
    K_hist = []
    for pos, vel in zip(pos_snaps, vel_snaps):
        v2 = np.sum(vel**2, axis=1)
        K_hist.append(0.5 * m_par * np.sum(v2))
    K_hist = np.array(K_hist)
    ax.plot(times, K_hist / K_hist[0], "o-", ms=4)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("t [Myr]"); ax.set_ylabel("K(t) / K(0)")
    ax.set_title("Kinetic energy evolution")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "stability_profile.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Stage 1] Stability figure: {out}")

    # Return scalar diagnostics
    pos_final = pos_snaps[-1]
    vel_final = vel_snaps[-1]
    n_par = len(pos_final)
    r_final = np.sqrt(np.sum(pos_final**2, axis=1))
    r_init  = np.sqrt(np.sum(pos_snaps[0]**2, axis=1))
    M_init  = enclosed_mass_particles(pos_snaps[0], m_par, np.array([R200]))[0]
    M_final = enclosed_mass_particles(pos_final, m_par, np.array([R200]))[0]

    K_final = K_hist[-1]
    K_init  = K_hist[0]

    return {
        "M_within_r200_initial": float(M_init),
        "M_within_r200_final":   float(M_final),
        "frac_M_r200_change":    float((M_final - M_init) / M_init),
        "COM_offset_final_kpc":  float(com_offsets[-1]),
        "K_ratio_final_over_init": float(K_final / K_init),
        "t_final_Myr":           float(times[-1]),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.quick:
        n_par   = 4096
        n_steps = 20
        print("[Stage 1] QUICK mode: n_par=4096, n_steps=20")
    elif args.orbit_demo:
        n_par = 4096
        n_steps = 240
        print("[Stage 1] ORBIT-DEMO mode: n_par=4096, n_steps=240")
    else:
        n_par   = args.n_par
        n_steps = args.n_steps

    n_grid  = args.n_grid
    r_max   = args.r_max if args.r_max else 2.5 * R200
    l_box   = args.l_box if args.l_box else 4.0 * R200
    dx      = l_box / n_grid
    max_dt  = args.max_dt

    print("=" * 60)
    print("Stage 1: Single-halo DM-only non-cosmological run")
    print("=" * 60)
    print(f"  N_par={n_par:,}  N_grid={n_grid}  n_steps={n_steps}")
    print(f"  L_box={l_box:.1f} kpc  dx={dx:.2f} kpc")
    print(f"  r_max={r_max:.0f} kpc  r200={R200:.0f} kpc")
    print(f"  max_dt={max_dt:.2f} Myr")
    print(f"  Output: {OUT_DIR}")

    # --------------- Build IC ---------------
    t0 = time.time()
    pos0, vel0, m_par, ref = build_ic(n_par, r_max)
    print(f"[Stage 1] IC generated in {time.time()-t0:.1f} s")

    # --------------- Validate IC ---------------
    ic_diags = validate_ic(pos0, vel0, m_par, r_max, ref)

    # Build PM force object (used in both IC plots and evolution)
    force = PhysicalDMGravityForce(n_grid, l_box, G=G, subtract_mean=True, cfl_ff=0.3)

    # --------------- IC plots ---------------
    print("[Stage 1] Generating IC diagnostic plots...")
    plot_ic_diagnostics(pos0, vel0, m_par, force, n_par, r_max, l_box, ref)

    # --------------- Evolution ---------------
    snapshot_every = args.snapshot_every if args.snapshot_every else max(1, n_steps // 10)
    pos_snaps, vel_snaps, times = run_evolution(
        pos0, vel0, m_par, n_grid, l_box, n_steps,
        snapshot_every=snapshot_every, max_dt=max_dt
    )
    traj_diags = plot_particle_trajectories(
        pos_snaps, vel_snaps, times, l_box, select_tracked_particles(pos0, vel0)
    )

    # --------------- Stability diagnostics ---------------
    stab_diags = plot_stability(pos_snaps, vel_snaps, times, m_par, r_max, ref)

    # --------------- Save params + scalars ---------------
    params_dict = {
        "stage": 1,
        "description": "DM-only single NFW halo, physical coordinates",
        "halo": {
            "M200_Msun": M200, "conc": CONC,
            "r200_kpc": float(R200), "a_scale_kpc": float(A_SCALE),
            "rho_s_Msun_per_kpc3": float(f"{RHOS:.4e}"),
            "redshift": Z,
        },
        "grid": {
            "N_grid": n_grid, "L_box_kpc": float(l_box),
            "dx_kpc": float(dx), "N_par": n_par,
            "r_max_kpc": float(r_max), "m_par_Msun": float(m_par),
        },
        "evolution": {
            "n_steps": n_steps, "snapshot_every": snapshot_every,
            "max_dt_Myr": float(max_dt),
            "t_final_Myr": float(times[-1]),
            "t_circ_a_scale_Myr": float(circular_period(A_SCALE)),
            "t_circ_0p2r200_Myr": float(circular_period(0.2 * R200)),
            "t_circ_r200_Myr": float(circular_period(R200)),
        },
    }
    with open(os.path.join(OUT_DIR, "stage1_params.yaml"), "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False, sort_keys=False)

    # Scalar summary
    all_rows = ["=" * 60, "Stage 1 Scalar Diagnostics", "=" * 60, ""]
    all_rows += ["--- IC validation ---"]
    for k, v in ic_diags.items():
        all_rows.append(f"  {k}: {v}")
    all_rows += ["", "--- Tracked trajectories ---"]
    for k, v in traj_diags.items():
        all_rows.append(f"  {k}: {v}")
    all_rows += ["", "--- Stability diagnostics ---"]
    for k, v in stab_diags.items():
        all_rows.append(f"  {k}: {v}")

    with open(os.path.join(OUT_DIR, "scalars.txt"), "w") as f:
        f.write("\n".join(all_rows) + "\n")

    print("\n[Stage 1] DONE. Files written:")
    for fname in sorted(os.listdir(OUT_DIR)):
        print(f"  merger/outputs/stage1/{fname}")


if __name__ == "__main__":
    main()
