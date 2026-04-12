"""
stage1_binary_dm_merger.py
==========================
DM-only binary cluster merger debug runner.

This isolates the dark matter dynamics from the gas by evolving two live DM
halos with only the PM (or optional direct) gravity force. It can use either:

  1. the exact GAMER-generated merger particle files, or
  2. the current DiffHydro-resampled halo ICs.

That makes it easier to debug why a merger remnant may stay as two clumps
instead of re-virializing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax.numpy as jnp

import diffhydro as dh
from diffhydro.equationmanager import EquationManager
from merger.physical_pm_force import PhysicalDMGravityForce
from merger.stage1_dm_halo import build_ic as build_dm_particle_ic, build_gas_state, R200, G, GAMMA
from merger.stage2_gas_dm_halo import DirectParticleHybridGravityForce


OUT_ROOT = os.path.join(_HERE, "outputs", "stage1_binary_dm_merger")
os.makedirs(OUT_ROOT, exist_ok=True)

GAMER_PAR1 = os.path.join(_HERE, "gamer_merger", "merger2_0_particles.h5")
GAMER_PAR2 = os.path.join(_HERE, "gamer_merger", "merger2_1_particles.h5")
L_BOX_DEFAULT = 12288.0
N_GRID_DEFAULT = 128
SEPARATION_DEFAULT = 4000.0
VREL_KMS_DEFAULT = 1000.0
KM_S_TO_KPC_MYR = 1.02269e-3


def parse_args():
    p = argparse.ArgumentParser(description="DM-only binary merger debug runner")
    p.add_argument("--ic-source", type=str, default="gamer", choices=["gamer", "resample"], help="Use GAMER particle files or resampled DiffHydro halo ICs")
    p.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT, help="PM mesh resolution")
    p.add_argument("--l-box", type=float, default=L_BOX_DEFAULT, help="Box size [kpc]")
    p.add_argument("--separation", type=float, default=SEPARATION_DEFAULT, help="Initial cluster separation [kpc]")
    p.add_argument("--vrel-kms", type=float, default=VREL_KMS_DEFAULT, help="Relative speed [km/s]")
    p.add_argument("--n-steps", type=int, default=240, help="Number of evolution steps")
    p.add_argument("--max-dt", type=float, default=10.0, help="Maximum timestep [Myr]")
    p.add_argument("--snapshot-every", type=int, default=12, help="Snapshot cadence")
    p.add_argument("--gravity-method", type=str, default="pm", choices=["pm", "direct"], help="DM gravity solver")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Softening for direct particle gravity")
    p.add_argument("--n-par-per-cluster", type=int, default=None, help="Optional particle count per cluster; for GAMER ICs this downsamples while preserving total mass")
    p.add_argument("--r-max", type=float, default=0.75 * R200, help="Resampled-halo truncation radius [kpc]")
    p.add_argument("--seed", type=int, default=1234, help="Seed for particle downsampling")
    p.add_argument("--output-subdir", type=str, default="gamer_ic_dm_only", help="Output subdirectory")
    p.add_argument("--quick", action="store_true", help="Short smoke test")
    return p.parse_args()


def compute_centers(l_box, separation):
    center = np.array([0.5 * l_box, 0.5 * l_box, 0.5 * l_box], dtype=np.float64)
    center1 = center.copy()
    center2 = center.copy()
    center1[0] -= 0.5 * separation
    center2[0] += 0.5 * separation
    return center1, center2


def load_gamer_particle_file(path, n_select=None, seed=1234):
    with h5py.File(path, "r") as f:
        pos = np.asarray(f["dm"]["particle_position"], dtype=np.float32)
        vel = np.asarray(f["dm"]["particle_velocity"], dtype=np.float32)
        m = np.asarray(f["dm"]["particle_mass"], dtype=np.float32)
    m_par = float(m[0])
    n_total = pos.shape[0]
    if n_select is not None and n_select < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=int(n_select), replace=False)
        pos = pos[idx]
        vel = vel[idx]
        m_par = m_par * (float(n_total) / float(n_select))
    return pos, vel, np.float32(m_par), n_total


def build_binary_dm_particles_from_gamer(n_par_per_cluster, l_box, separation, vrel_kms, seed):
    center1, center2 = compute_centers(l_box, separation)
    v_half = 0.5 * float(vrel_kms) * KM_S_TO_KPC_MYR
    bulk1 = np.array([+v_half, 0.0, 0.0], dtype=np.float32)
    bulk2 = np.array([-v_half, 0.0, 0.0], dtype=np.float32)

    pos1, vel1, m_par1, n_full1 = load_gamer_particle_file(GAMER_PAR1, n_par_per_cluster, seed)
    pos2, vel2, m_par2, n_full2 = load_gamer_particle_file(GAMER_PAR2, n_par_per_cluster, seed + 1)
    if not np.isclose(m_par1, m_par2):
        raise ValueError("Cluster particle masses differ unexpectedly.")

    pos_box_1 = np.mod(pos1 + center1[None, :], l_box).astype(np.float32)
    pos_box_2 = np.mod(pos2 + center2[None, :], l_box).astype(np.float32)
    vel_box_1 = (vel1 + bulk1[None, :]).astype(np.float32)
    vel_box_2 = (vel2 + bulk2[None, :]).astype(np.float32)
    labels = np.concatenate([np.zeros(pos_box_1.shape[0], dtype=np.int32), np.ones(pos_box_2.shape[0], dtype=np.int32)])

    return {
        "pos": np.concatenate([pos_box_1, pos_box_2], axis=0),
        "vel": np.concatenate([vel_box_1, vel_box_2], axis=0),
        "m_par": np.float32(m_par1),
        "labels": labels,
        "n_par_total": int(labels.size),
        "n_par_cluster_1": int(pos_box_1.shape[0]),
        "n_par_cluster_2": int(pos_box_2.shape[0]),
        "n_full_cluster_1": int(n_full1),
        "n_full_cluster_2": int(n_full2),
        "center1": center1.astype(np.float32),
        "center2": center2.astype(np.float32),
        "v_half": float(v_half),
    }


def build_binary_dm_particles_resampled(n_par_per_cluster, r_max, l_box, separation, vrel_kms):
    center1, center2 = compute_centers(l_box, separation)
    v_half = 0.5 * float(vrel_kms) * KM_S_TO_KPC_MYR
    bulk1 = np.array([+v_half, 0.0, 0.0], dtype=np.float32)
    bulk2 = np.array([-v_half, 0.0, 0.0], dtype=np.float32)

    pos1, vel1, m_par1, _ = build_dm_particle_ic(n_par_per_cluster, r_max)
    pos2, vel2, m_par2, _ = build_dm_particle_ic(n_par_per_cluster, r_max)
    if not np.isclose(m_par1, m_par2):
        raise ValueError("Cluster particle masses differ unexpectedly.")

    pos_box_1 = np.mod(pos1 + center1[None, :], l_box).astype(np.float32)
    pos_box_2 = np.mod(pos2 + center2[None, :], l_box).astype(np.float32)
    vel_box_1 = (vel1 + bulk1[None, :]).astype(np.float32)
    vel_box_2 = (vel2 + bulk2[None, :]).astype(np.float32)
    labels = np.concatenate([np.zeros(pos_box_1.shape[0], dtype=np.int32), np.ones(pos_box_2.shape[0], dtype=np.int32)])

    return {
        "pos": np.concatenate([pos_box_1, pos_box_2], axis=0),
        "vel": np.concatenate([vel_box_1, vel_box_2], axis=0),
        "m_par": np.float32(m_par1),
        "labels": labels,
        "n_par_total": int(labels.size),
        "n_par_cluster_1": int(pos_box_1.shape[0]),
        "n_par_cluster_2": int(pos_box_2.shape[0]),
        "center1": center1.astype(np.float32),
        "center2": center2.astype(np.float32),
        "v_half": float(v_half),
    }


def run_dm_only_evolution(dm_ic, n_grid, l_box, n_steps, snapshot_every, max_dt, gravity_method="pm", softening_cells=0.5):
    eq = EquationManager()
    eq.gamma = GAMMA
    eq.mesh_shape = [n_grid, n_grid, n_grid]
    eq.box_size = (l_box, l_box, l_box)

    if gravity_method == "direct":
        force = DirectParticleHybridGravityForce(
            n_grid,
            l_box,
            G=G,
            subtract_mean=True,
            cfl_ff=0.3,
            include_gas_in_gravity=False,
            softening_cells=softening_cells,
        )
    else:
        force = PhysicalDMGravityForce(
            n_grid,
            l_box,
            G=G,
            subtract_mean=True,
            cfl_ff=0.3,
            include_gas_in_gravity=False,
        )

    solver = dh.HLLC(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov)
    flux = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="VANLEER"), positivity=True)
    flux.dx_o = l_box / n_grid

    U_curr = build_gas_state(n_grid, l_box).astype(np.float32)
    params_curr = {
        "dm": {
            "pos": np.asarray(dm_ic["pos"], dtype=np.float32),
            "vel": np.asarray(dm_ic["vel"], dtype=np.float32),
            "m_par": np.asarray(dm_ic["m_par"], dtype=np.float32),
        }
    }
    dm_pos_snaps = [np.asarray(params_curr["dm"]["pos"], dtype=np.float32)]
    dm_vel_snaps = [np.asarray(params_curr["dm"]["vel"], dtype=np.float32)]

    sims = {}
    steps_done = 0
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
        U_curr, params_curr = sims[n_chunk].evolve(U_curr, params_curr)
        steps_done += n_chunk
        dm_pos_snaps.append(np.asarray(params_curr["dm"]["pos"], dtype=np.float32))
        dm_vel_snaps.append(np.asarray(params_curr["dm"]["vel"], dtype=np.float32))
        print(f"  step {steps_done}/{n_steps}", flush=True)

    t_wall = time.time() - t_wall_start
    t_myr = np.arange(len(dm_pos_snaps), dtype=np.float64) * float(snapshot_every) * float(max_dt)
    return dm_pos_snaps, dm_vel_snaps, t_myr, t_wall


def minimal_image(delta, l_box):
    return (delta + 0.5 * l_box) % l_box - 0.5 * l_box


def compute_snapshot_metrics(dm_pos_snaps, dm_vel_snaps, labels, l_box):
    metrics = []
    labels = np.asarray(labels)
    for pos, vel in zip(dm_pos_snaps, dm_vel_snaps):
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        mask1 = labels == 0
        mask2 = labels == 1
        pos1 = pos[mask1]
        pos2 = pos[mask2]
        vel1 = vel[mask1]
        vel2 = vel[mask2]
        com1 = np.mean(pos1, axis=0)
        com2 = np.mean(pos2, axis=0)
        dcom = minimal_image(com2 - com1, l_box)
        sep = float(np.linalg.norm(dcom))
        bulk_rel = float(np.linalg.norm(np.mean(vel2, axis=0) - np.mean(vel1, axis=0)))
        all_pos = pos.copy()
        all_com = np.mean(all_pos, axis=0)
        rad = np.sqrt(np.sum(minimal_image(all_pos - all_com[None, :], l_box) ** 2, axis=1))
        metrics.append({
            "com_sep_kpc": sep,
            "bulk_rel_speed_kpc_myr": bulk_rel,
            "median_radius_kpc": float(np.median(rad)),
            "p90_radius_kpc": float(np.percentile(rad, 90.0)),
        })
    return metrics


def write_metric_history(t_myr, metrics, out_dir):
    arrs = {
        "t_myr": np.asarray(t_myr, dtype=np.float32),
        "com_sep_kpc": np.asarray([m["com_sep_kpc"] for m in metrics], dtype=np.float32),
        "bulk_rel_speed_kpc_myr": np.asarray([m["bulk_rel_speed_kpc_myr"] for m in metrics], dtype=np.float32),
        "median_radius_kpc": np.asarray([m["median_radius_kpc"] for m in metrics], dtype=np.float32),
        "p90_radius_kpc": np.asarray([m["p90_radius_kpc"] for m in metrics], dtype=np.float32),
    }
    out = os.path.join(out_dir, "metric_history.npz")
    np.savez(out, **arrs)
    print(f"  Saved {out}")


def plot_com_separation(t_myr, metrics, out_dir):
    sep = [m["com_sep_kpc"] for m in metrics]
    vrel = [m["bulk_rel_speed_kpc_myr"] for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_myr, sep, "o-")
    axes[0].set_title("Origin-tagged COM separation")
    axes[0].set_xlabel("t [Myr]")
    axes[0].set_ylabel("separation [kpc]")
    axes[0].grid(ls=":", alpha=0.5)
    axes[1].plot(t_myr, vrel, "o-")
    axes[1].set_title("Origin-tagged bulk relative speed")
    axes[1].set_xlabel("t [Myr]")
    axes[1].set_ylabel("relative speed [kpc/Myr]")
    axes[1].grid(ls=":", alpha=0.5)
    out = os.path.join(out_dir, "dm_com_separation_history.png")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


def make_dm_density_animation(dm_pos_snaps, t_myr, m_par, n_grid, l_box, out_dir):
    force = PhysicalDMGravityForce(
        n_grid,
        l_box,
        G=G,
        subtract_mean=True,
        include_gas_in_gravity=False,
    )
    rho_slices = []
    for pos in dm_pos_snaps:
        rho = np.asarray(force._deposit_density(jnp.asarray(pos, dtype=jnp.float32), jnp.asarray(m_par, dtype=jnp.float32)))
        rho_slices.append(rho[:, :, n_grid // 2])

    log_slices = [np.log10(np.maximum(sl, 1.0e-30)) for sl in rho_slices]
    vmin = min(float(np.min(sl)) for sl in log_slices)
    vmax = max(float(np.max(sl)) for sl in log_slices)
    L = l_box / 2.0
    extent = [-L, L, -L, L]
    frames = []
    for t_now, log_sl in zip(t_myr, log_slices):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(log_sl.T, origin="lower", extent=extent, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(rf"$\log_{{10}}\rho_{{DM}}$ at $t={t_now:.0f}$ Myr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    out = os.path.join(out_dir, "dm_density_evolution.gif")
    imageio.mimsave(out, frames, duration=0.6, loop=0)
    print(f"  Saved {out}")


def main():
    args = parse_args()
    if args.quick:
        args.n_steps = 24
        args.snapshot_every = 4
        args.max_dt = 10.0
        if args.n_par_per_cluster is None:
            args.n_par_per_cluster = 4096 if args.ic_source == "gamer" else 2048

    out_dir = os.path.join(OUT_ROOT, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if args.ic_source == "gamer":
        if not os.path.exists(GAMER_PAR1) or not os.path.exists(GAMER_PAR2):
            raise FileNotFoundError("Missing GAMER particle IC files in merger/gamer_merger.")
        dm_ic = build_binary_dm_particles_from_gamer(
            args.n_par_per_cluster,
            float(args.l_box),
            float(args.separation),
            float(args.vrel_kms),
            int(args.seed),
        )
    else:
        n_par = int(args.n_par_per_cluster) if args.n_par_per_cluster is not None else 32768
        dm_ic = build_binary_dm_particles_resampled(
            n_par,
            float(args.r_max),
            float(args.l_box),
            float(args.separation),
            float(args.vrel_kms),
        )

    print(
        f"[DM Binary] IC source={args.ic_source}, gravity={args.gravity_method}\n"
        f"  N_total={dm_ic['n_par_total']:,}, m_par={float(dm_ic['m_par']):.3e} Msun\n"
        f"  centers: c1={dm_ic['center1']}, c2={dm_ic['center2']}\n"
        f"  v_half={dm_ic['v_half']:.4f} kpc/Myr"
    )
    print("[DM Binary] Evolving on GPU...")
    dm_pos_snaps, dm_vel_snaps, t_myr, t_wall = run_dm_only_evolution(
        dm_ic,
        int(args.n_grid),
        float(args.l_box),
        int(args.n_steps),
        int(args.snapshot_every),
        float(args.max_dt),
        gravity_method=str(args.gravity_method),
        softening_cells=float(args.softening_cells),
    )
    print(f"[DM Binary] Done. Wall time={t_wall:.1f} s")

    print("[DM Binary] Writing diagnostics...")
    metrics = compute_snapshot_metrics(dm_pos_snaps, dm_vel_snaps, dm_ic["labels"], float(args.l_box))
    plot_com_separation(t_myr, metrics, out_dir)
    make_dm_density_animation(dm_pos_snaps, t_myr, dm_ic["m_par"], int(args.n_grid), float(args.l_box), out_dir)
    write_metric_history(t_myr, metrics, out_dir)

    com_sep_hist = np.asarray([m["com_sep_kpc"] for m in metrics], dtype=np.float64)
    finite = np.isfinite(com_sep_hist)
    if np.any(finite):
        idx_min = int(np.nanargmin(com_sep_hist))
        min_sep = float(com_sep_hist[idx_min])
        t_min_sep = float(t_myr[idx_min])
    else:
        idx_min = -1
        min_sep = float("nan")
        t_min_sep = float("nan")

    scalars = {
        "ic_source": str(args.ic_source),
        "n_grid": int(args.n_grid),
        "l_box_kpc": float(args.l_box),
        "dx_kpc": float(args.l_box) / int(args.n_grid),
        "separation_kpc": float(args.separation),
        "vrel_kms": float(args.vrel_kms),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "n_steps": int(args.n_steps),
        "max_dt_Myr": float(args.max_dt),
        "snapshot_every": int(args.snapshot_every),
        "n_par_total": int(dm_ic["n_par_total"]),
        "n_par_cluster_1": int(dm_ic["n_par_cluster_1"]),
        "n_par_cluster_2": int(dm_ic["n_par_cluster_2"]),
        "m_par_Msun": float(dm_ic["m_par"]),
        "t_final_Myr_est": float(t_myr[-1]),
        "t_wall_s": float(t_wall),
        "com_sep_initial_kpc": float(metrics[0]["com_sep_kpc"]),
        "com_sep_final_kpc": float(metrics[-1]["com_sep_kpc"]),
        "com_sep_min_kpc": float(min_sep),
        "com_sep_min_time_Myr": float(t_min_sep),
        "bulk_rel_speed_initial_kpc_myr": float(metrics[0]["bulk_rel_speed_kpc_myr"]),
        "bulk_rel_speed_final_kpc_myr": float(metrics[-1]["bulk_rel_speed_kpc_myr"]),
        "median_radius_final_kpc": float(metrics[-1]["median_radius_kpc"]),
        "p90_radius_final_kpc": float(metrics[-1]["p90_radius_kpc"]),
    }
    with open(os.path.join(out_dir, "scalars.yaml"), "w") as fh:
        yaml.safe_dump(scalars, fh, sort_keys=False)
    print(f"  Saved {os.path.join(out_dir, 'scalars.yaml')}")


if __name__ == "__main__":
    main()
