"""
stage2_binary_cluster_merger.py
===============================
DiffHydro binary cluster merger matching the GAMER head-on merger ICs in
``merger/gamer_merger/gen_ics.py``.

Setup:
  - two equal clusters, M200 = 5e14 Msun
  - separation = 4000 kpc
  - relative velocity = 1000 km/s
  - box size = 12288 kpc
  - default mesh = 128^3 (dx = 96 kpc)
  - field hydro + live DM particles
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import merger.stage2_gas_dm_halo as s2
from merger.stage1_dm_halo import build_ic as build_dm_particle_ic, R200
from merger.stage1_binary_dm_merger import build_binary_dm_particles_from_gamer


OUT_ROOT = os.path.join(_HERE, "outputs", "stage2_binary_merger")
os.makedirs(OUT_ROOT, exist_ok=True)

PROFILE1 = os.path.join(_HERE, "gamer_merger", "profile1_gamer.h5")
PROFILE2 = os.path.join(_HERE, "gamer_merger", "profile2_gamer.h5")
L_BOX_DEFAULT = 12288.0
N_GRID_DEFAULT = 128
SEPARATION_DEFAULT = 4000.0
VREL_KMS_DEFAULT = 1000.0
KM_S_TO_KPC_MYR = 1.02269e-3


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2-style DiffHydro binary merger matching the GAMER merger ICs")
    p.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT, help="Hydro mesh resolution")
    p.add_argument("--l-box", type=float, default=L_BOX_DEFAULT, help="Box size [kpc]")
    p.add_argument("--separation", type=float, default=SEPARATION_DEFAULT, help="Cluster separation [kpc]")
    p.add_argument("--vrel-kms", type=float, default=VREL_KMS_DEFAULT, help="Relative approach speed [km/s]")
    p.add_argument("--n-steps", type=int, default=240, help="Number of evolution steps")
    p.add_argument("--max-dt", type=float, default=20.0, help="Maximum timestep [Myr]")
    p.add_argument("--snapshot-every", type=int, default=12, help="Snapshot cadence")
    p.add_argument("--dm-mode", type=str, default="particles", choices=["particles", "static"], help="Live DM particles or fixed DM field")
    p.add_argument("--ic-source", type=str, default="gamer", choices=["gamer", "resample"], help="Use exact GAMER DM particle ICs or resampled DiffHydro halos")
    p.add_argument("--n-par-per-cluster", type=int, default=32768, help="DM particles per cluster when --dm-mode=particles")
    p.add_argument("--r-max", type=float, default=0.75 * R200, help="DM particle truncation radius [kpc]")
    p.add_argument("--seed", type=int, default=1234, help="Seed for downsampling GAMER particle ICs")
    p.add_argument("--gravity-method", type=str, default="pm", choices=["pm", "direct"], help="DM gravity method for live particles")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Softening for direct live-DM particle forces")
    p.add_argument("--output-subdir", type=str, default="gamer_like_headon", help="Output subdirectory")
    p.add_argument("--quick", action="store_true", help="Short smoke test")
    return p.parse_args()


def load_profile(profile_h5):
    with h5py.File(profile_h5, "r") as f:
        g = f["fields"]
        return {
            "radius": np.asarray(g["radius"], dtype=np.float64),
            "rho_g": np.asarray(g["density"], dtype=np.float64),
            "p_g": np.asarray(g["pressure"], dtype=np.float64),
            "rho_dm": np.asarray(g["dark_matter_density"], dtype=np.float64),
        }


def grid_coordinates(n_grid, l_box):
    dx = l_box / n_grid
    x = (np.arange(n_grid, dtype=np.float64) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, dx


def shifted_profile_to_grid(profile, X, Y, Z, center):
    rr = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    radius = profile["radius"]
    return {
        "rho_g": np.interp(rr.ravel(), radius, profile["rho_g"], left=profile["rho_g"][0], right=profile["rho_g"][-1]).reshape(rr.shape),
        "p_g": np.interp(rr.ravel(), radius, profile["p_g"], left=profile["p_g"][0], right=profile["p_g"][-1]).reshape(rr.shape),
        "rho_dm": np.interp(rr.ravel(), radius, profile["rho_dm"], left=profile["rho_dm"][0], right=profile["rho_dm"][-1]).reshape(rr.shape),
    }


def compute_centers(l_box, separation):
    center = np.array([0.5 * l_box, 0.5 * l_box, 0.5 * l_box], dtype=np.float64)
    center1 = center.copy()
    center2 = center.copy()
    center1[0] -= 0.5 * separation
    center2[0] += 0.5 * separation
    return center1, center2


def build_binary_gas_state(profile1, profile2, n_grid, l_box, center1, center2, v1, v2):
    X, Y, Z, _ = grid_coordinates(n_grid, l_box)
    c1 = shifted_profile_to_grid(profile1, X, Y, Z, center1)
    c2 = shifted_profile_to_grid(profile2, X, Y, Z, center2)

    rho_bg = max(float(profile1["rho_g"][-1]), float(profile2["rho_g"][-1]), 1.0e-12)
    p_bg = max(float(profile1["p_g"][-1]), float(profile2["p_g"][-1]), 1.0e-14)

    rho1_exc = np.maximum(c1["rho_g"] - rho_bg, 0.0)
    rho2_exc = np.maximum(c2["rho_g"] - rho_bg, 0.0)
    p1_exc = np.maximum(c1["p_g"] - p_bg, 0.0)
    p2_exc = np.maximum(c2["p_g"] - p_bg, 0.0)

    rho = rho_bg + rho1_exc + rho2_exc
    mx = rho1_exc * v1[0] + rho2_exc * v2[0]
    my = rho1_exc * v1[1] + rho2_exc * v2[1]
    mz = rho1_exc * v1[2] + rho2_exc * v2[2]
    p = p_bg + p1_exc + p2_exc
    e_th = p / (s2.GAMMA - 1.0)
    e_kin = 0.5 * rho1_exc * np.dot(v1, v1) + 0.5 * rho2_exc * np.dot(v2, v2)
    E = e_th + e_kin
    U0 = np.stack([rho, mx, my, mz, E], axis=0).astype(np.float32)

    rho_dm = np.maximum(c1["rho_dm"] + c2["rho_dm"], 0.0).astype(np.float32)
    meta = {
        "rho_bg": float(rho_bg),
        "p_bg": float(p_bg),
        "center1": center1.astype(np.float32),
        "center2": center2.astype(np.float32),
    }
    return U0, rho_dm, meta


def build_binary_dm_particles_resampled(n_par_per_cluster, r_max, l_box, center1, center2, v1, v2):
    pos1, vel1, m_par1, _ = build_dm_particle_ic(n_par_per_cluster, r_max)
    pos2, vel2, m_par2, _ = build_dm_particle_ic(n_par_per_cluster, r_max)
    if not np.isclose(m_par1, m_par2):
        raise ValueError("Cluster particle masses differ unexpectedly.")

    pos_box_1 = (pos1 + center1[None, :]).astype(np.float32)
    pos_box_2 = (pos2 + center2[None, :]).astype(np.float32)
    vel_box_1 = (vel1 + v1[None, :]).astype(np.float32)
    vel_box_2 = (vel2 + v2[None, :]).astype(np.float32)

    pos = np.mod(np.concatenate([pos_box_1, pos_box_2], axis=0), l_box).astype(np.float32)
    vel = np.concatenate([vel_box_1, vel_box_2], axis=0).astype(np.float32)
    return {
        "pos": pos,
        "vel": vel,
        "m_par": np.float32(m_par1),
        "n_par_total": int(pos.shape[0]),
    }


def density_slice_animation(U_snaps, t_myr, n_grid, l_box, out_dir):
    L = l_box / 2.0
    extent = [-L, L, -L, L]
    rho_slices = [np.asarray(U[0])[:, :, n_grid // 2] for U in U_snaps]
    log_slices = [np.log10(np.maximum(sl, 1.0e-30)) for sl in rho_slices]
    vmin = min(float(np.min(sl)) for sl in log_slices)
    vmax = max(float(np.max(sl)) for sl in log_slices)
    frames = []

    for t_now, log_sl in zip(t_myr, log_slices):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(log_sl.T, origin="lower", extent=extent, cmap="plasma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(rf"$\log_{{10}}\rho_g$ at $t={t_now:.0f}$ Myr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    out = os.path.join(out_dir, "gas_density_evolution.gif")
    imageio.mimsave(out, frames, duration=0.6, loop=0)
    print(f"  Saved {out}")


def plot_merger_slices(U_snaps, t_myr, n_grid, l_box, out_dir):
    sl0 = np.asarray(U_snaps[0][0])[:, :, n_grid // 2]
    slf = np.asarray(U_snaps[-1][0])[:, :, n_grid // 2]
    L = l_box / 2.0
    extent = [-L, L, -L, L]
    vmin = min(float(np.min(np.log10(np.maximum(sl0, 1.0e-30)))), float(np.min(np.log10(np.maximum(slf, 1.0e-30)))))
    vmax = max(float(np.max(np.log10(np.maximum(sl0, 1.0e-30)))), float(np.max(np.log10(np.maximum(slf, 1.0e-30)))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, sl, t_now, tag in [
        (axes[0], sl0, t_myr[0], "Initial"),
        (axes[1], slf, t_myr[-1], "Final"),
    ]:
        im = ax.imshow(np.log10(np.maximum(sl, 1.0e-30)).T, origin="lower", extent=extent, cmap="plasma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{tag} gas density, t={t_now:.0f} Myr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    out = os.path.join(out_dir, "gas_density_slices.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_velocity_history(U_snaps, t_myr, out_dir):
    vmax = []
    mach95 = []
    for U in U_snaps:
        rho = np.maximum(np.asarray(U[0]), 1.0e-30)
        mx, my, mz = np.asarray(U[1]), np.asarray(U[2]), np.asarray(U[3])
        vx = mx / rho
        vy = my / rho
        vz = mz / rho
        kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        p = (s2.GAMMA - 1.0) * np.maximum(np.asarray(U[4]) - kinetic, 1.0e-30)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        cs = np.sqrt(s2.GAMMA * p / rho)
        vmax.append(float(np.max(speed)))
        mach95.append(float(np.percentile(speed / np.maximum(cs, 1.0e-30), 95.0)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_myr, vmax, "o-")
    axes[0].set_title("Peak gas speed")
    axes[0].set_xlabel("t [Myr]")
    axes[0].set_ylabel("v_max [kpc/Myr]")
    axes[0].grid(ls=":", alpha=0.5)
    axes[1].plot(t_myr, mach95, "o-")
    axes[1].set_title("Mach95")
    axes[1].set_xlabel("t [Myr]")
    axes[1].set_ylabel("Mach")
    axes[1].grid(ls=":", alpha=0.5)
    out = os.path.join(out_dir, "velocity_history.png")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


def main():
    args = parse_args()
    if args.quick:
        args.n_steps = 24
        args.snapshot_every = 4
        args.max_dt = 15.0
        if args.dm_mode == "particles":
            args.n_par_per_cluster = min(int(args.n_par_per_cluster), 4096)

    out_dir = os.path.join(OUT_ROOT, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(PROFILE1) or not os.path.exists(PROFILE2):
        raise FileNotFoundError("Missing merger/gamer_merger profile files. Run merger/gamer_merger/gen_ics.py first.")

    profile1 = load_profile(PROFILE1)
    profile2 = load_profile(PROFILE2)
    center1, center2 = compute_centers(float(args.l_box), float(args.separation))
    v_half = 0.5 * float(args.vrel_kms) * KM_S_TO_KPC_MYR
    v1 = np.array([+v_half, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([-v_half, 0.0, 0.0], dtype=np.float32)

    print("[Binary Merger] Building hydro and DM ICs...")
    U0, rho_dm, meta = build_binary_gas_state(profile1, profile2, int(args.n_grid), float(args.l_box), center1, center2, v1, v2)
    dm_particles = None
    if args.dm_mode == "particles":
        if args.ic_source == "gamer":
            dm_particles = build_binary_dm_particles_from_gamer(
                int(args.n_par_per_cluster),
                float(args.l_box),
                float(args.separation),
                float(args.vrel_kms),
                int(args.seed),
            )
        else:
            dm_particles = build_binary_dm_particles_resampled(
                int(args.n_par_per_cluster),
                float(args.r_max),
                float(args.l_box),
                center1,
                center2,
                v1,
                v2,
            )

    print(
        f"  centers: c1={center1}, c2={center2}\n"
        f"  v1={v1} kpc/Myr, v2={v2} kpc/Myr\n"
        f"  gas peak rho={float(np.max(U0[0])):.3e} Msun/kpc^3"
    )
    if dm_particles is not None:
        print(
            f"  live DM: source={args.ic_source}, "
            f"N_total={dm_particles['n_par_total']:,}, "
            f"m_par={float(dm_particles['m_par']):.3e} Msun"
        )

    print("[Binary Merger] Evolving on GPU...")
    s2.OUT_DIR = out_dir
    U_snaps, t_myr, t_wall, dm_pos_snaps = s2.run_evolution(
        U0,
        rho_dm,
        int(args.n_grid),
        float(args.l_box),
        int(args.n_steps),
        int(args.snapshot_every),
        float(args.max_dt),
        dm_particles=dm_particles,
        gravity_method=args.gravity_method,
        softening_cells=float(args.softening_cells),
    )
    print(f"[Binary Merger] Done. Wall time={t_wall:.1f} s")

    print("[Binary Merger] Writing diagnostics...")
    plot_merger_slices(U_snaps, t_myr, int(args.n_grid), float(args.l_box), out_dir)
    density_slice_animation(U_snaps, t_myr, int(args.n_grid), float(args.l_box), out_dir)
    plot_velocity_history(U_snaps, t_myr, out_dir)
    if dm_particles is not None:
        s2.plot_dm_profile_evolution(dm_pos_snaps, t_myr, dm_particles["m_par"], int(args.n_grid), float(args.l_box))
        s2.make_dm_density_animation(dm_pos_snaps, t_myr, dm_particles["m_par"], int(args.n_grid), float(args.l_box), out_dir)

    scalars = {
        "n_grid": int(args.n_grid),
        "l_box_kpc": float(args.l_box),
        "dx_kpc": float(args.l_box) / int(args.n_grid),
        "separation_kpc": float(args.separation),
        "vrel_kms": float(args.vrel_kms),
        "v_cluster_kpc_myr": float(v_half),
        "n_steps": int(args.n_steps),
        "max_dt_Myr": float(args.max_dt),
        "snapshot_every": int(args.snapshot_every),
        "t_final_Myr_est": float(t_myr[-1]),
        "t_wall_s": float(t_wall),
        "dm_mode": str(args.dm_mode),
        "ic_source": str(args.ic_source),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "n_par_per_cluster": int(args.n_par_per_cluster) if dm_particles is not None else 0,
        "n_par_total": int(dm_particles["n_par_total"]) if dm_particles is not None else 0,
        "m_par_Msun": float(dm_particles["m_par"]) if dm_particles is not None else 0.0,
        "center1_kpc": [float(x) for x in center1],
        "center2_kpc": [float(x) for x in center2],
        "rho_bg_Msun_kpc3": float(meta["rho_bg"]),
        "p_bg_Msun_kpc_Myr2": float(meta["p_bg"]),
        "rho_peak_initial": float(np.max(np.asarray(U_snaps[0][0]))),
        "rho_peak_final": float(np.max(np.asarray(U_snaps[-1][0]))),
    }
    with open(os.path.join(out_dir, "scalars.yaml"), "w") as fh:
        yaml.safe_dump(scalars, fh, sort_keys=False)
    print(f"  Saved {os.path.join(out_dir, 'scalars.yaml')}")


if __name__ == "__main__":
    main()
