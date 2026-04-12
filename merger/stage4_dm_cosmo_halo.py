"""
stage4_dm_cosmo_halo.py
=======================
Stage 4: DM-only cosmological evolution of a static isolated halo.

This script ports the Stage 1 single-halo DM test into the repository's
supercomoving/JaxPM stepping conventions. Gas is kept dynamically inert so the
diagnostics isolate DM momentum conventions, Hubble expansion, and PM gravity.
"""

from __future__ import annotations

import copy
import argparse
import os
import sys
import time
import yaml
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

import diffhydro as dh
from diffhydro.cosmology import (
    BackgroundExpansionForce,
    JaxPMCoupledGravityForce,
    LCDMBackground,
    SuperComovingEquationManager,
)
from diffhydro.cosmology import conversions as cosmo_conv
from diffhydro.equationmanager import EquationManager

from merger.halo_reference import HaloReference


OUT_DIR = os.path.join(_HERE, "outputs", "stage4")
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA = 5.0 / 3.0
Z_INIT = 0.295
M200 = 5.0e14
CONC = 3.5
H0_KM_S_MPC = 70.0
OMEGA_M = 0.3
OMEGA_B = 0.0486
OMEGA_L = 0.7
KM_S_TO_KPC_MYR = 1.0227121650537077e-3
G_PHYS = 4.3009172706e-6 * KM_S_TO_KPC_MYR**2


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4 DM-only cosmological single-halo validation")
    p.add_argument("--n-par", type=int, default=4096, help="Number of DM particles")
    p.add_argument("--n-grid", type=int, default=32, help="PM mesh resolution")
    p.add_argument("--a-final", type=float, default=1.0, help="Final scale factor")
    p.add_argument("--max-steps", type=int, default=240, help="Maximum number of KDK steps")
    p.add_argument("--snapshot-every", type=int, default=20, help="Snapshot cadence in steps")
    p.add_argument("--max-dtau", type=float, default=100.0, help="Maximum supercomoving step size")
    p.add_argument("--min-dtau", type=float, default=1.0e-3, help="Minimum supercomoving step size")
    p.add_argument("--dtau-safety", type=float, default=0.8, help="Safety factor multiplying the remaining-a timestep estimate")
    p.add_argument("--dm-kick-scale", type=float, default=1.0, help="Optional rescaling of the DM kick factor")
    p.add_argument("--velocity-scale", type=float, default=0.10, help="Optional rescaling of the initial peculiar velocities")
    p.add_argument("--gravity-method", type=str, default="direct", choices=["jaxpm", "direct"], help="DM gravity solver for Stage 4")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Plummer-like softening length for direct gravity, in comoving cell units")
    p.add_argument("--output-subdir", type=str, default=None, help="Optional run subdirectory inside merger/outputs/stage4")
    p.add_argument("--r-max", type=float, default=None, help="Particle truncation radius [kpc]")
    p.add_argument("--l-box", type=float, default=None, help="Initial physical box size [kpc]")
    return p.parse_args()



def primitives_from_U(eq, U):
    W = np.asarray(eq.get_primitives_from_conservatives(jnp.asarray(U)))
    return {
        "rho": W[0],
        "vx": W[1],
        "vy": W[2],
        "vz": W[3],
        "p": W[4],
    }


def temperature_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30)

def plot_stage2_slices(eq, U_snaps):
    prim_i = primitives_from_U(eq, U_snaps[0])
    prim_f = primitives_from_U(eq, U_snaps[-1])

    rho_i = prim_i["rho"]
    p_i = prim_i["p"]
    T_i = temperature_proxy(prim_i["rho"], prim_i["p"])
    rho_f = prim_f["rho"]
    p_f = prim_f["p"]
    T_f = temperature_proxy(prim_f["rho"], prim_f["p"])

    N = rho_i.shape[0]
    sl_i = [
        np.log10(rho_i[:, :, N // 2] + 1e-30),
        np.log10(p_i[:, :, N // 2] + 1e-30),
        np.log10(T_i[:, :, N // 2] + 1e-30),
    ]
    sl_f = [
        np.log10(rho_f[:, :, N // 2] + 1e-30),
        np.log10(p_f[:, :, N // 2] + 1e-30),
        np.log10(T_f[:, :, N // 2] + 1e-30),
    ]

    titles = [r"$\log_{10}\rho_g$", r"$\log_{10}p$", r"$\log_{10}T_{\rm proxy}$"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j in range(3):
        vmin = min(np.min(sl_i[j]), np.min(sl_f[j]))
        vmax = max(np.max(sl_i[j]), np.max(sl_f[j]))
        im = axes[0, j].imshow(sl_i[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"Initial {titles[j]}")
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
        im = axes[1, j].imshow(sl_f[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"Final {titles[j]}")
        plt.colorbar(im, ax=axes[1, j], fraction=0.046)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "slices_initial_final.png")
    plt.savefig(out, dpi=140)
    plt.close()

    frac_rho = (rho_f[:, :, N // 2] - rho_i[:, :, N // 2]) / np.maximum(rho_i[:, :, N // 2], 1.0e-30)
    frac_p = (p_f[:, :, N // 2] - p_i[:, :, N // 2]) / np.maximum(p_i[:, :, N // 2], 1.0e-30)
    frac_T = (T_f[:, :, N // 2] - T_i[:, :, N // 2]) / np.maximum(np.abs(T_i[:, :, N // 2]), 1.0e-30)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    frac_fields = [frac_rho, frac_p, frac_T]
    frac_titles = [r"$\Delta\rho/\rho$", r"$\Delta p/p$", r"$\Delta T/T$"]
    for ax, arr, title in zip(axes, frac_fields, frac_titles):
        vmax = np.nanpercentile(np.abs(arr), 99.0)
        im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "slices_fractional_change.png")
    plt.savefig(out, dpi=140)
    plt.close()

def hubble_to_myr_inv(h0_km_s_mpc: float) -> float:
    return float(h0_km_s_mpc) * KM_S_TO_KPC_MYR / 1000.0


class DirectNBodyCosmoForce:
    """Direct-summation DM force in the Stage 4 supercomoving convention."""

    def __init__(self, *, bg_h0, dx_com, n_grid, softening_cells=0.5, cfl_acc=0.5):
        self.bg_h0 = jnp.asarray(bg_h0, dtype=jnp.float32)
        self.dx_com = jnp.asarray(dx_com, dtype=jnp.float32)
        self.n_grid = float(n_grid)
        self.softening_cells = jnp.asarray(softening_cells, dtype=jnp.float32)
        self.cfl_acc = float(cfl_acc)

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def force(self, i, U_gas, params, dtau):
        del i
        dtau = jnp.maximum(jnp.asarray(dtau), 0.0)
        dtau_half = 0.5 * dtau

        params_out = dict(params)
        a = jnp.asarray(params_out.get("a", 1.0), dtype=jnp.float32)
        dm_params = dict(params_out["dm"])

        x = jnp.asarray(dm_params["x"], dtype=jnp.float32)
        p = jnp.asarray(dm_params["p_or_v"], dtype=jnp.float32)
        m = jnp.asarray(dm_params.get("m_par", dm_params.get("mass", 1.0)), dtype=jnp.float32)
        if m.ndim == 0:
            m = jnp.ones((x.shape[0],), dtype=jnp.float32) * m

        accel_p_old = self._canonical_accel(x, m, a)
        p_half = p + dtau_half * accel_p_old
        x_drift = jnp.mod(x + dtau * self.bg_h0 * p_half, self.n_grid)

        accel_p_new = self._canonical_accel(x_drift, m, a)
        p_new = p_half + dtau_half * accel_p_new

        dm_params["x"] = x_drift
        dm_params["p_or_v"] = p_new
        params_out["dm"] = dm_params
        return U_gas, params_out

    def _canonical_accel(self, x_cells, masses, a):
        delta = x_cells[:, None, :] - x_cells[None, :, :]
        delta = (delta + 0.5 * self.n_grid) % self.n_grid - 0.5 * self.n_grid
        r2_cells = jnp.sum(delta * delta, axis=-1)
        eps2 = self.softening_cells * self.softening_cells
        mask = 1.0 - jnp.eye(x_cells.shape[0], dtype=jnp.float32)
        inv_r3 = mask * jax.lax.rsqrt(jnp.maximum(r2_cells + eps2, 1.0e-12)) ** 3

        # Physical acceleration:
        # g = -G m r_phys / (|r_phys|^2 + eps_phys^2)^(3/2)
        #   = -G m delta_cells / [(a*dx_com)^2 * (r_cells^2 + eps_cells^2)^(3/2)]
        length2 = jnp.maximum((a * self.dx_com) ** 2, 1.0e-20)
        pair = -G_PHYS * masses[None, :, None] * delta * (inv_r3[..., None] / length2)
        g_phys = jnp.sum(pair, axis=1)
        return (a * a / (self.dx_com * self.bg_h0)) * g_phys


def make_sim(eq, bg, mesh_shape, *, gravity_method="jaxpm", dx_com=None, softening_cells=0.5):
    solver = dh.HLLC(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov)
    flux = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="VANLEER"), positivity=True)
    if dx_com is None:
        raise ValueError("dx_com is required for Stage 4 hydro flux spacing")
    flux.dx_o = float(dx_com)
    bg_force = BackgroundExpansionForce(bg, a_init=1.0)
    if gravity_method == "direct":
        if dx_com is None:
            raise ValueError("dx_com is required for gravity_method='direct'")
        grav_force = DirectNBodyCosmoForce(
            bg_h0=bg.H0,
            dx_com=dx_com,
            n_grid=mesh_shape[0],
            softening_cells=softening_cells,
        )
    else:
        grav_force = JaxPMCoupledGravityForce(
            eq,
            mesh_shape=mesh_shape,
            subtract_mean=True,
            use_jaxpm=True,
            dm_drift_factor=bg.H0,
            dm_kick_factor=1.0,
            gas_kick_factor=0.0,
        )
    return dh.hydro(
        n_super_step=1,
        max_dt=0.2,
        fluxes=[flux],
        forces=[bg_force, grav_force],
        use_mol=True,
        pmesh_shape=(1, 1, 1),
        dx_o=float(dx_com),
    )


def build_initial_state(ref, args):
    a_init = 1.0 / (1.0 + Z_INIT)
    r_max = args.r_max if args.r_max else 2.5 * ref.r200
    l_box_phys = args.l_box if args.l_box else 4.0 * ref.r200
    l_box_com = l_box_phys / a_init
    dx_phys = l_box_phys / args.n_grid
    dx_com = l_box_com / args.n_grid

    bg = LCDMBackground(
        h=H0_KM_S_MPC / 100.0,
        Omega_m=OMEGA_M,
        Omega_b=OMEGA_B,
        Omega_lambda=OMEGA_L,
        Omega_k=0.0,
        use_jax_cosmo=True,
    )
    bg.H0 = hubble_to_myr_inv(H0_KM_S_MPC)

    base_eq = EquationManager()
    base_eq.gamma = GAMMA
    base_eq.mesh_shape = [args.n_grid, args.n_grid, args.n_grid]
    base_eq.box_size = (l_box_com, l_box_com, l_box_com)
    eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)

    rho_floor_phys = np.full((args.n_grid, args.n_grid, args.n_grid), 1.0e-6, dtype=np.float32)
    p_floor_phys = np.full_like(rho_floor_phys, 1.0e-8)
    zeros = np.zeros_like(rho_floor_phys)
    w_code = cosmo_conv.primitives_phys_to_code(
        jnp.asarray(rho_floor_phys),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
        jnp.asarray(p_floor_phys),
        jnp.asarray(a_init, dtype=jnp.float32),
    )
    U0 = eq.get_conservatives_from_primitives(w_code)

    np.random.seed(42)
    particles = ref.hse.generate_dm_particles(args.n_par, r_max=r_max)
    pos_phys = np.array(particles[("dm", "particle_position")], dtype=np.float32)
    vel_phys = np.array(particles[("dm", "particle_velocity")], dtype=np.float32) * np.float32(args.velocity_scale)
    m_par = float(np.array(particles[("dm", "particle_mass")])[0])

    center_phys = np.full(3, 0.5 * l_box_phys, dtype=np.float32)
    center_cells = center_phys / dx_phys
    pos_cells = (pos_phys + center_phys[None, :]) / dx_phys
    vel_tilde_cells = (a_init * vel_phys) / dx_com
    p_or_v = vel_tilde_cells / bg.H0

    params0 = {
        "a": jnp.asarray(a_init, dtype=jnp.float32),
        "dm": {
            "x": jnp.asarray(pos_cells, dtype=jnp.float32),
            "p_or_v": jnp.asarray(p_or_v, dtype=jnp.float32),
            "mass": jnp.ones((args.n_par,), dtype=jnp.float32),
            "m_par": jnp.asarray(m_par, dtype=jnp.float32),
            "drift_factor": jnp.asarray(bg.H0, dtype=jnp.float32),
            "kick_factor": jnp.asarray(1.5 * OMEGA_M * bg.H0 * a_init * args.dm_kick_scale, dtype=jnp.float32),
            "gas_kick_factor": jnp.asarray(0.0, dtype=jnp.float32),
        },
    }
    meta = {
        "a_init": float(a_init),
        "r_max": float(r_max),
        "l_box_phys": float(l_box_phys),
        "l_box_com": float(l_box_com),
        "dx_phys": float(dx_phys),
        "dx_com": float(dx_com),
        "center_cells": center_cells.astype(np.float32),
        "m_par": float(m_par),
        "pos_phys_init": pos_phys,
        "vel_phys_init": vel_phys,
    }
    return eq, bg, U0, params0, meta


def minimal_image(delta_cells, n_grid):
    return (delta_cells + 0.5 * n_grid) % n_grid - 0.5 * n_grid


def snapshot_from_params(params, meta, n_grid, a_init):
    a = float(params["a"])
    pos_cells = np.asarray(params["dm"]["x"], dtype=np.float64)
    p_or_v = np.asarray(params["dm"]["p_or_v"], dtype=np.float64)
    delta_cells = minimal_image(pos_cells - meta["center_cells"][None, :], n_grid)
    pos_com = delta_cells * meta["dx_com"]
    pos_phys = pos_com * a
    vel_tilde_cells = p_or_v * (meta["bg_h0"])
    vel_phys = vel_tilde_cells * meta["dx_com"] / a
    r_phys = np.sqrt(np.sum(pos_phys**2, axis=1))
    r_com = np.sqrt(np.sum(pos_com**2, axis=1))
    vr_phys = np.sum(pos_phys * vel_phys, axis=1) / np.maximum(r_phys, 1.0e-30)
    vmag = np.sqrt(np.sum(vel_phys**2, axis=1))
    return {
        "a": a,
        "step": int(meta["step"]),
        "pos_phys": pos_phys,
        "pos_com": pos_com,
        "vel_phys": vel_phys,
        "r_phys": r_phys,
        "r_com": r_com,
        "vr_phys": vr_phys,
        "vmag": vmag,
        "p_or_v": p_or_v,
    }


def spherical_potential_energy(r_phys, m_par):
    r_sorted = np.sort(np.asarray(r_phys, dtype=np.float64))
    m_prev = m_par * np.arange(r_sorted.size, dtype=np.float64)
    return -np.sum(G_PHYS * m_par * m_prev / np.maximum(r_sorted, 1.0e-30))

def compute_snapshot_metrics(snap, m_par, ref, a_init):
    pos = snap["pos_phys"]
    vel = snap["vel_phys"]
    pos_com = snap["pos_com"]
    com_com = np.mean(pos_com, axis=0)
    vel_com = np.mean(vel, axis=0)
    pos_rel = pos - np.mean(pos, axis=0)[None, :]
    vel_rel = vel - vel_com[None, :]

    K = 0.5 * m_par * np.sum(np.sum(vel_rel**2, axis=1))
    U = spherical_potential_energy(np.sqrt(np.sum(pos_rel**2, axis=1)), m_par)
    L_vec = m_par * np.sum(np.cross(pos_rel, vel_rel), axis=0)

    r_phys = snap["r_phys"]
    r_com = snap["r_com"]
    mass_r200_phys = m_par * np.sum(r_phys < ref.r200)
    mass_r200_com = m_par * np.sum(r_com < (ref.r200 / a_init))
    mass_r500_phys = m_par * np.sum(r_phys < ref.r500)
    mass_r500_com = m_par * np.sum(r_com < (ref.r500 / a_init))

    return {
        "a": float(snap["a"]),
        "step": int(snap["step"]),
        "virial_ratio": float(2.0 * K / max(abs(U), 1.0e-30)),
        "kinetic_energy": float(K),
        "potential_energy": float(U),
        "com_drift_com_kpc": float(np.linalg.norm(com_com)),
        "bulk_speed_kpc_myr": float(np.linalg.norm(vel_com)),
        "angular_momentum_norm": float(np.linalg.norm(L_vec)),
        "mass_r200_phys": float(mass_r200_phys),
        "mass_r200_com": float(mass_r200_com),
        "mass_r500_phys": float(mass_r500_phys),
        "mass_r500_com": float(mass_r500_com),
        "median_radius_phys": float(np.median(r_phys)),
        "median_speed_phys": float(np.median(np.sqrt(np.sum(vel**2, axis=1)))),
    }


def density_profile(r, m_par, r_bins):
    shell_mass, _ = np.histogram(r, bins=r_bins, weights=np.full(r.shape, m_par))
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    return shell_mass / np.maximum(shell_vol, 1.0e-30)


def enclosed_mass(r, m_par, r_eval):
    r_sorted = np.sort(r)
    counts = np.searchsorted(r_sorted, r_eval, side="right")
    return counts.astype(np.float64) * m_par


def plot_profiles(ref, snapshots, m_par, out_dir):
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.5 * ref.r200), 50)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    select_idx = np.unique(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots))).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].loglog(ref.r_grid, ref.rho_dm, "k--", lw=2, label="DM target")
    for idx in select_idx:
        snap = snapshots[idx]
        rho = density_profile(snap["r_phys"], m_par, r_bins)
        axes[0].loglog(r_c, rho, label=f"a={snap['a']:.3f}")
    axes[0].set_xlabel("r [kpc]")
    axes[0].set_ylabel(r"$\rho_{\rm dm}$ [$M_\odot$/kpc$^3$]")
    axes[0].set_title("Density profile evolution")
    axes[0].legend(fontsize=8)

    axes[1].loglog(ref.r_grid, np.sqrt(G_PHYS * np.maximum(ref.mass_dm, 0.0) / np.maximum(ref.r_grid, 1.0e-30)), "k--", lw=2, label="DM target")
    for idx in select_idx:
        snap = snapshots[idx]
        m_enc = enclosed_mass(snap["r_phys"], m_par, r_c)
        vc = np.sqrt(G_PHYS * m_enc / np.maximum(r_c, 1.0e-30))
        axes[1].loglog(r_c, vc, label=f"a={snap['a']:.3f}")
    axes[1].set_xlabel("r [kpc]")
    axes[1].set_ylabel(r"$v_c$ [kpc/Myr]")
    axes[1].set_title("Circular-velocity curve")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profiles_multi_a.png"), dpi=140)
    plt.close()


def plot_phase_space(snapshots, ref, out_dir):
    select_idx = np.unique(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots))).astype(int))
    fig, axes = plt.subplots(1, len(select_idx), figsize=(4.0 * len(select_idx), 4.0), sharey=True)
    if len(select_idx) == 1:
        axes = [axes]
    for ax, idx in zip(axes, select_idx):
        snap = snapshots[idx]
        ax.scatter(snap["r_phys"] / ref.r200, snap["vr_phys"], s=3, alpha=0.35, rasterized=True)
        ax.axhline(0.0, color="k", lw=1)
        ax.axvline(1.0, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r/r_{200}$")
        ax.set_title(f"a={snap['a']:.3f}")
    axes[0].set_ylabel(r"$v_r$ [kpc/Myr]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_space_snapshots.png"), dpi=140)
    plt.close()


def plot_tracked_trajectories(snapshots, ref, out_dir, n_track=8, seed=123):
    if len(snapshots) < 2:
        return

    n_par = snapshots[0]["pos_phys"].shape[0]
    rng = np.random.default_rng(seed)
    track_ids = np.sort(rng.choice(n_par, size=min(n_track, n_par), replace=False))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(track_ids)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    final_pos = snapshots[-1]["pos_phys"]
    axes[0].scatter(final_pos[:, 0], final_pos[:, 1], s=2, alpha=0.15, color="0.5", rasterized=True)
    axes[0].add_patch(plt.Circle((0.0, 0.0), ref.r200, fill=False, ls="--", lw=1.0, color="k", alpha=0.8))
    for color, pid in zip(colors, track_ids):
        traj = np.array([snap["pos_phys"][pid] for snap in snapshots])
        axes[0].plot(traj[:, 0], traj[:, 1], "-", lw=1.8, color=color)
        axes[0].plot(traj[0, 0], traj[0, 1], "o", ms=4, color=color)
        axes[0].plot(traj[-1, 0], traj[-1, 1], "s", ms=4, color=color)
    axes[0].set_title("Late-time particles with tracked trajectories")
    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].set_aspect("equal", adjustable="box")

    a_hist = [snap["a"] for snap in snapshots]
    for color, pid in zip(colors, track_ids):
        radial_track = np.array([snap["r_phys"][pid] for snap in snapshots])
        axes[1].plot(a_hist, radial_track / ref.r200, "-o", lw=1.6, ms=3.5, color=color)
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_title("Tracked particle radii")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel(r"$r/r_{200}$")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tracked_trajectories.png"), dpi=140)
    plt.close()


def make_dm_projection_animation(snapshots, ref, out_dir):
    if len(snapshots) < 2:
        return
    r_lim = 1.6 * ref.r200
    bins = 128
    frames = []
    hist_list = []
    for snap in snapshots:
        x = snap["pos_phys"][:, 0]
        y = snap["pos_phys"][:, 1]
        hist, _, _ = np.histogram2d(x, y, bins=bins, range=[[-r_lim, r_lim], [-r_lim, r_lim]])
        hist_list.append(hist.T)
    log_list = [np.log10(h + 1.0e-3) for h in hist_list]
    vmin = min(float(np.min(h)) for h in log_list)
    vmax = max(float(np.max(h)) for h in log_list)
    extent = [-r_lim, r_lim, -r_lim, r_lim]

    for snap, hist, logh in zip(snapshots, hist_list, log_list):
        frac = (hist - hist_list[0]) / np.maximum(hist_list[0], 1.0)
        lim = max(0.1, float(np.nanpercentile(np.abs(frac), 99.0)))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        im0 = axes[0].imshow(logh, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        axes[0].set_title(rf"DM projection  a={snap['a']:.3f}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(frac, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal")
        axes[1].set_title("Fractional projected change")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]")
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(os.path.join(out_dir, "dm_projection_evolution.gif"), frames, duration=0.8, loop=0)


def plot_time_series(history, out_dir):
    a = np.array([h["a"] for h in history])
    vir = np.array([h["virial_ratio"] for h in history])
    com = np.array([h["com_drift_com_kpc"] for h in history])
    ang = np.array([h["angular_momentum_norm"] for h in history])
    m200_phys = np.array([h["mass_r200_phys"] for h in history])
    m200_com = np.array([h["mass_r200_com"] for h in history])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(a, vir, "o-")
    axes[0, 0].set_title(r"Virial ratio $2K/|U|$")
    axes[0, 0].set_xlabel("a")

    axes[0, 1].plot(a, com, "o-")
    axes[0, 1].set_title("COM drift in comoving frame")
    axes[0, 1].set_xlabel("a")
    axes[0, 1].set_ylabel("kpc")

    axes[1, 0].plot(a, ang / max(ang[0], 1.0e-30), "o-")
    axes[1, 0].axhline(1.0, color="k", ls="--", lw=1)
    axes[1, 0].set_title("Angular momentum drift")
    axes[1, 0].set_xlabel("a")
    axes[1, 0].set_ylabel(r"$|L|/|L_0|$")

    axes[1, 1].plot(a, m200_phys / max(m200_phys[0], 1.0e-30), "o-", label=r"$M(<r_{200,\rm phys,0})$")
    axes[1, 1].plot(a, m200_com / max(m200_com[0], 1.0e-30), "s-", label=r"$M(<r_{200,\rm com,0})$")
    axes[1, 1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1, 1].set_title("Aperture-mass evolution")
    axes[1, 1].set_xlabel("a")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_series_metrics.png"), dpi=140)
    plt.close()


def write_snapshot_bundle(history, out_dir):
    np.savez(
        os.path.join(out_dir, "snapshot_history.npz"),
        a=np.array([h["a"] for h in history], dtype=np.float32),
        step=np.array([h["step"] for h in history], dtype=np.int32),
        virial_ratio=np.array([h["virial_ratio"] for h in history], dtype=np.float32),
        com_drift_com_kpc=np.array([h["com_drift_com_kpc"] for h in history], dtype=np.float32),
        angular_momentum_norm=np.array([h["angular_momentum_norm"] for h in history], dtype=np.float32),
        mass_r200_phys=np.array([h["mass_r200_phys"] for h in history], dtype=np.float32),
        mass_r200_com=np.array([h["mass_r200_com"] for h in history], dtype=np.float32),
        mass_r500_phys=np.array([h["mass_r500_phys"] for h in history], dtype=np.float32),
        mass_r500_com=np.array([h["mass_r500_com"] for h in history], dtype=np.float32),
    )


def main():
    global OUT_DIR
    args = parse_args()
    if args.output_subdir:
        OUT_DIR = os.path.join(_HERE, "outputs", "stage4", args.output_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = HaloReference(z=Z_INIT, m200_msun=M200, conc=CONC)
    eq, bg, U_in, params, meta = build_initial_state(ref, args)
    U = copy.deepcopy(U_in)
    meta["bg_h0"] = float(bg.H0)
    sim = make_sim(
        eq,
        bg,
        tuple(eq.mesh_shape),
        gravity_method=args.gravity_method,
        dx_com=meta["dx_com"],
        softening_cells=args.softening_cells,
    )

    a_init = meta["a_init"]
    a_final = max(float(args.a_final), a_init + 1.0e-5)

    snapshots = []
    history = []

    meta["step"] = 0
    snap0 = snapshot_from_params(params, meta, args.n_grid, a_init)
    snapshots.append(snap0)
    history.append(compute_snapshot_metrics(snap0, meta["m_par"], ref, a_init))

    wall_t0 = time.time()
    step_i = 0
    while float(params["a"]) < a_final:
        if step_i >= args.max_steps:
            raise RuntimeError(
                f"Reached max_steps={args.max_steps} before a_final={a_final:.4f}. "
                "Increase --max-steps or --max-dtau."
            )
        a_now = float(params["a"])
        da_dtau = float(bg.da_dtau(a_now))
        remaining = max(a_final - a_now, 0.0)
        dtau = min(float(args.max_dtau), float(args.dtau_safety) * remaining / max(da_dtau, 1.0e-12))
        dtau = max(dtau, float(args.min_dtau))

        params_dm = dict(params["dm"])
        params_dm["drift_factor"] = jnp.asarray(bg.H0, dtype=jnp.float32)
        params_dm["kick_factor"] = jnp.asarray(1.5 * OMEGA_M * bg.H0 * a_now * args.dm_kick_scale, dtype=jnp.float32)
        params_dm["gas_kick_factor"] = jnp.asarray(0.0, dtype=jnp.float32)
        params = dict(params)
        params["dm"] = params_dm

        U, params = sim._hydrostep(step_i, (U, params), jnp.asarray(dtau, dtype=jnp.float32))
        step_i += 1

        if (step_i % args.snapshot_every) == 0 or float(params["a"]) >= a_final:
            meta["step"] = step_i
            snap = snapshot_from_params(params, meta, args.n_grid, a_init)
            snapshots.append(snap)
            history.append(compute_snapshot_metrics(snap, meta["m_par"], ref, a_init))
    
    wall = time.time() - wall_t0

    plot_profiles(ref, snapshots, meta["m_par"], OUT_DIR)
    plot_phase_space(snapshots, ref, OUT_DIR)
    plot_tracked_trajectories(snapshots, ref, OUT_DIR)
    make_dm_projection_animation(snapshots, ref, OUT_DIR)
    plot_stage2_slices(eq,[U_in,U])
    plot_time_series(history, OUT_DIR)
    write_snapshot_bundle(history, OUT_DIR)

    prof_bins = np.logspace(np.log10(0.05 * ref.r200), np.log10(ref.r200), 32)
    rho_init = density_profile(snapshots[0]["r_phys"], meta["m_par"], prof_bins)
    rho_final = density_profile(snapshots[-1]["r_phys"], meta["m_par"], prof_bins)
    valid = (rho_init > 0.0) & (rho_final > 0.0)
    profile_drift = float(np.mean(np.abs(np.log(rho_final[valid] / rho_init[valid])))) if np.any(valid) else np.nan

    scalars = {
        "a_init": float(a_init),
        "a_final": float(history[-1]["a"]),
        "z_init": float(Z_INIT),
        "steps_taken": int(step_i),
        "snapshot_count": int(len(snapshots)),
        "wall_time_s": float(wall),
        "h0_myr_inv": float(bg.H0),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "velocity_scale": float(args.velocity_scale),
        "dm_kick_scale": float(args.dm_kick_scale),
        "virial_ratio_init": float(history[0]["virial_ratio"]),
        "virial_ratio_final": float(history[-1]["virial_ratio"]),
        "profile_drift_logmean": float(profile_drift),
        "com_drift_final_com_kpc": float(history[-1]["com_drift_com_kpc"]),
        "angular_momentum_ratio_final": float(history[-1]["angular_momentum_norm"] / max(history[0]["angular_momentum_norm"], 1.0e-30)),
        "mass_r200_phys_ratio_final": float(history[-1]["mass_r200_phys"] / max(history[0]["mass_r200_phys"], 1.0e-30)),
        "mass_r200_com_ratio_final": float(history[-1]["mass_r200_com"] / max(history[0]["mass_r200_com"], 1.0e-30)),
        "mass_r500_phys_ratio_final": float(history[-1]["mass_r500_phys"] / max(history[0]["mass_r500_phys"], 1.0e-30)),
        "mass_r500_com_ratio_final": float(history[-1]["mass_r500_com"] / max(history[0]["mass_r500_com"], 1.0e-30)),
        "median_radius_phys_ratio_final": float(history[-1]["median_radius_phys"] / max(history[0]["median_radius_phys"], 1.0e-30)),
        "median_speed_phys_ratio_final": float(history[-1]["median_speed_phys"] / max(history[0]["median_speed_phys"], 1.0e-30)),
    }
    params_out = {
        "stage": 4,
        "description": "Single-halo DM-only cosmological evolution in supercomoving variables",
        "halo": {
            "z_init": float(Z_INIT),
            "M200_Msun": float(M200),
            "conc": float(CONC),
            "r200_kpc": float(ref.r200),
            "r500_kpc": float(ref.r500),
            "r_max_kpc": float(meta["r_max"]),
        },
        "cosmology": {
            "H0_km_s_Mpc": float(H0_KM_S_MPC),
            "H0_Myr_inv": float(bg.H0),
            "Omega_m": float(OMEGA_M),
            "Omega_b": float(OMEGA_B),
            "Omega_lambda": float(OMEGA_L),
            "a_init": float(a_init),
            "a_final_target": float(a_final),
        },
        "run": {
            "n_grid": int(args.n_grid),
            "n_par": int(args.n_par),
            "l_box_phys_kpc": float(meta["l_box_phys"]),
            "l_box_com_kpc": float(meta["l_box_com"]),
            "dx_phys_kpc": float(meta["dx_phys"]),
            "dx_com_kpc": float(meta["dx_com"]),
            "max_steps": int(args.max_steps),
            "snapshot_every": int(args.snapshot_every),
            "max_dtau": float(args.max_dtau),
            "min_dtau": float(args.min_dtau),
            "dtau_safety": float(args.dtau_safety),
            "gravity_method": str(args.gravity_method),
            "softening_cells": float(args.softening_cells),
            "dm_kick_scale": float(args.dm_kick_scale),
            "velocity_scale": float(args.velocity_scale),
        },
    }

    with open(os.path.join(OUT_DIR, "stage4_params.yaml"), "w") as f:
        yaml.dump(params_out, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(OUT_DIR, "scalars.txt"), "w") as f:
        f.write("Stage 4 Scalar Diagnostics\n")
        for k, v in scalars.items():
            f.write(f"{k}: {v}\n")

    print("[Stage 4] Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
