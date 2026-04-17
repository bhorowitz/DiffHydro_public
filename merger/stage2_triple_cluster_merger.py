"""
stage2_triple_cluster_merger.py
===============================
Static-background DiffHydro three-cluster merger following the configuration in
``merger/cluster_gen_gamer_run6.py`` while using the same non-cosmological
Stage 2 hydro+live-DM evolution path as ``stage2_binary_cluster_merger.py``.
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
import cluster_generator as cg
from astropy.cosmology import FlatLambdaCDM
from astropy import units as us

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import merger.stage2_gas_dm_halo as s2
from merger.halo_gas_ic import load_model_profile_tables


OUT_ROOT = os.path.join(_HERE, "outputs", "stage2_triple_merger")
os.makedirs(OUT_ROOT, exist_ok=True)

Z = 0.295
COS = FlatLambdaCDM(H0=70.0, Om0=0.3)
RHOC = COS.critical_density(Z).to(us.M_sun / us.kpc**3).value
KM_S_TO_KPC_MYR = 1.02269e-3

M200_1 = 5.0e14
M200_2 = 5.0e14
M200_3 = 1.0e14 #former 1.0E14
CONC1 = 3.5
CONC2 = 3.5
CONC3 = 4.2

L_BOX_DEFAULT = 15000.0
N_GRID_DEFAULT = 128
MAX_DT_DEFAULT = 20.0
NSTEPS_DEFAULT = 600
SNAPSHOT_EVERY_DEFAULT = 12


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 static three-cluster merger matching cluster_gen_gamer_run6.py")
    p.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT, help="Hydro mesh resolution")
    p.add_argument("--l-box", type=float, default=L_BOX_DEFAULT, help="Box size [kpc]")
    p.add_argument("--n-steps", type=int, default=NSTEPS_DEFAULT, help="Number of evolution steps")
    p.add_argument("--max-dt", type=float, default=MAX_DT_DEFAULT, help="Maximum timestep [Myr]")
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY_DEFAULT, help="Snapshot cadence")
    p.add_argument("--dm-mode", type=str, default="particles", choices=["particles", "static"], help="Live DM particles or fixed DM field")
    p.add_argument("--n-par-main", type=int, default=32768, help="DM particles for each 5e14 Msun main cluster")
    p.add_argument("--n-par-3", type=int, default=None, help="DM particles for the 1e14 Msun subcluster (default scales with mass)")
    p.add_argument("--r-max", type=float, default=5000.0, help="Particle truncation radius [kpc]")
    p.add_argument("--seed", type=int, default=24, help="Base seed for DM particle generation")
    p.add_argument("--gravity-method", type=str, default="pm", choices=["pm", "direct"], help="DM gravity method for live particles")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Softening for direct live-DM particle forces")
    p.add_argument("--output-subdir", type=str, default="run6_tt", help="Output subdirectory")
    p.add_argument("--quick", action="store_true", help="Short smoke test")
    return p.parse_args()


def build_cluster_model(m200, conc, gas_rc_frac, gas_rs_frac, gas_beta, gas_eps, gas_alpha, *, star_fraction=0.02):
    r200 = cg.find_overdensity_radius(m200, 200.0, z=Z)
    a = r200 / conc
    rhos = 200.0 * RHOC * conc**3 / ((np.log(1.0 + conc) - conc / (1.0 + conc)) * 3.0)
    mt = cg.nfw_mass_profile(rhos, a)
    r500, m500 = cg.find_radius_mass(mt, z=Z, delta=500.0)
    r2500, _ = cg.find_radius_mass(mt, z=Z, delta=2500.0)
    rhot = cg.nfw_density_profile(rhos, a)
    rho_star = star_fraction * rhot
    f_g = cg.f_gas(m500)
    rhog = cg.vikhlinin_density_profile(1.0, gas_rc_frac * r2500, gas_rs_frac * r200, gas_beta, gas_eps, gas_alpha)
    rhog = cg.rescale_profile_by_mass(rhog, f_g * m500, r500)
    hse = cg.HydrostaticEquilibrium.from_dens_and_tden(0.1, 20000.0, rhog, rhot, stellar_density=rho_star)
    prof = load_model_profile_tables(hse)
    return {
        "model": hse,
        "profile": prof,
        "r200": float(r200),
        "m200": float(m200),
    }



def cluster_run6_setup():
    cluster1 = build_cluster_model(M200_1, CONC1, 0.4, 0.8, 1.0, 0.8, 1.5)
    cluster2 = build_cluster_model(M200_2, CONC2, 0.2, 0.67, 1.0, 0.8, 1.5)
    cluster3 = build_cluster_model(M200_3, CONC3, 0.05, 0.15, 1.0, 0.8, 1.5)

    w = L_BOX_DEFAULT
    center1 = np.array([0.48 * w, 0.55 * w, 0.5 * w], dtype=np.float64)
    center2 = np.array([0.52 * w, 0.45 * w, 0.5 * w], dtype=np.float64)
    center3 = np.array([0.25 * w, 0.50 * w, 0.5 * w], dtype=np.float64) #offset from the main merger axis to make it more interesting...

    velocity = 1000.0 * KM_S_TO_KPC_MYR
    velocity1 = np.array([0.0, -0.5 * velocity * M200_2 / (M200_1 + M200_2), 0.0], dtype=np.float32)
    velocity2 = np.array([0.0, +0.5 * velocity * M200_1 / (M200_1 + M200_2), 0.0], dtype=np.float32)
    velocity3 = np.array([0.5 * velocity, 0.25 * velocity, 0.0], dtype=np.float32)

    return [
        {"name": "cluster1", "center": center1, "velocity": velocity1, **cluster1},
        {"name": "cluster2", "center": center2, "velocity": velocity2, **cluster2},
        {"name": "cluster3", "center": center3, "velocity": velocity3, **cluster3},
    ]

def cluster_run6_setup_play(): #ben's tweaked version with slightly different centers and velocities
    cluster1 = build_cluster_model(M200_1, CONC1, 0.4, 0.8, 1.0, 0.8, 1.5)
    cluster2 = build_cluster_model(M200_2, CONC2, 0.2, 0.67, 1.0, 0.8, 1.5)
    cluster3 = build_cluster_model(M200_3, 1.5*CONC3, 0.05, 0.15, 1.0, 0.8, 1.5)

    w = L_BOX_DEFAULT
    center1 = np.array([0.48 * w, 0.55 * w, 0.5 * w], dtype=np.float64)
    center2 = np.array([0.52 * w, 0.45 * w, 0.5 * w], dtype=np.float64)
    center3 = np.array([0.25 * w, 0.40 * w, 0.5 * w], dtype=np.float64) #offset from the main merger axis to make it more interesting...

    velocity = 1000.0 * KM_S_TO_KPC_MYR
    velocity1 = np.array([0.0, -0.5 * velocity * M200_2 / (M200_1 + M200_2), 0.0], dtype=np.float32)
    velocity2 = np.array([0.0, +0.5 * velocity * M200_1 / (M200_1 + M200_2), 0.0], dtype=np.float32)
    velocity3 = np.array([1.0 * velocity, 0.0 * velocity, 0.0], dtype=np.float32)

    return [
        {"name": "cluster1", "center": center1, "velocity": velocity1, **cluster1},
        {"name": "cluster2", "center": center2, "velocity": velocity2, **cluster2},
        {"name": "cluster3", "center": center3, "velocity": velocity3, **cluster3},
    ]


def grid_coordinates(n_grid, l_box):
    dx = float(l_box) / int(n_grid)
    x = (np.arange(n_grid, dtype=np.float64) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, dx


def shifted_profile_to_grid(profile, X, Y, Z, center):
    rr = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    radius = profile["radius"]
    return {
        "rho_g": np.interp(rr.ravel(), radius, profile["gas_density"], left=profile["gas_density"][0], right=profile["gas_density"][-1]).reshape(rr.shape),
        "p_g": np.interp(rr.ravel(), radius, profile["pressure"], left=profile["pressure"][0], right=profile["pressure"][-1]).reshape(rr.shape),
        "rho_dm": np.interp(rr.ravel(), radius, profile["dark_matter_density"], left=profile["dark_matter_density"][0], right=profile["dark_matter_density"][-1]).reshape(rr.shape),
    }


def build_triple_gas_state(clusters, n_grid, l_box):
    X, Y, Z, _ = grid_coordinates(n_grid, l_box)
    mapped = [shifted_profile_to_grid(cl["profile"], X, Y, Z, cl["center"]) for cl in clusters]

    rho_bg = max(float(cl["profile"]["gas_density"][-1]) for cl in clusters)
    p_bg = max(float(cl["profile"]["pressure"][-1]) for cl in clusters)
    rho_bg = max(rho_bg, 1.0e-12)
    p_bg = max(p_bg, 1.0e-14)

    rho = np.full((n_grid, n_grid, n_grid), rho_bg, dtype=np.float64)
    p = np.full((n_grid, n_grid, n_grid), p_bg, dtype=np.float64)
    mx = np.zeros_like(rho)
    my = np.zeros_like(rho)
    mz = np.zeros_like(rho)
    e_kin = np.zeros_like(rho)
    rho_dm = np.zeros_like(rho)

    for cl, m in zip(clusters, mapped):
        rho_exc = np.maximum(m["rho_g"] - rho_bg, 0.0)
        p_exc = np.maximum(m["p_g"] - p_bg, 0.0)
        rho += rho_exc
        p += p_exc
        mx += rho_exc * cl["velocity"][0]
        my += rho_exc * cl["velocity"][1]
        mz += rho_exc * cl["velocity"][2]
        e_kin += 0.5 * rho_exc * float(np.dot(cl["velocity"], cl["velocity"]))
        rho_dm += np.maximum(m["rho_dm"], 0.0)

    e_th = p / (s2.GAMMA - 1.0)
    U0 = np.stack([rho, mx, my, mz, e_th + e_kin], axis=0).astype(np.float32)
    meta = {
        "rho_bg": float(rho_bg),
        "p_bg": float(p_bg),
        "centers": [cl["center"].astype(np.float32) for cl in clusters],
    }
    return U0, rho_dm.astype(np.float32), meta


def build_triple_dm_particles(clusters, n_pars, r_max, l_box, seed):
    pos_all = []
    vel_all = []
    mass_all = []
    for i, (cl, n_par) in enumerate(zip(clusters, n_pars)):
        np.random.seed(int(seed) + i)
        part = cl["model"].generate_dm_particles(int(n_par), r_max=float(r_max))
        pos = np.asarray(part[("dm", "particle_position")], dtype=np.float32) + cl["center"][None, :].astype(np.float32)
        vel = np.asarray(part[("dm", "particle_velocity")], dtype=np.float32) + cl["velocity"][None, :]
        m_par = np.asarray(part[("dm", "particle_mass")], dtype=np.float32)
        pos_all.append(np.mod(pos, l_box).astype(np.float32))
        vel_all.append(vel.astype(np.float32))
        mass_all.append(m_par.astype(np.float32))
    return {
        "pos": np.concatenate(pos_all, axis=0).astype(np.float32),
        "vel": np.concatenate(vel_all, axis=0).astype(np.float32),
        "m_par": np.concatenate(mass_all, axis=0).astype(np.float32),
        "n_par_total": int(sum(int(n) for n in n_pars)),
    }


def density_slice_animation(U_snaps, t_myr, n_grid, l_box, out_dir):
    half = 0.5 * l_box
    extent = [-half, half, -half, half]
    log_slices = [np.log10(np.maximum(np.asarray(U[0])[:, :, n_grid // 2], 1.0e-30)) for U in U_snaps]
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
    imageio.mimsave(os.path.join(out_dir, "gas_density_evolution.gif"), frames, duration=0.6, loop=0)


def plot_density_slices(U_snaps, t_myr, n_grid, l_box, out_dir):
    half = 0.5 * l_box
    extent = [-half, half, -half, half]
    sl0 = np.log10(np.maximum(np.asarray(U_snaps[0][0])[:, :, n_grid // 2], 1.0e-30))
    slf = np.log10(np.maximum(np.asarray(U_snaps[-1][0])[:, :, n_grid // 2], 1.0e-30))
    vmin = min(float(np.min(sl0)), float(np.min(slf)))
    vmax = max(float(np.max(sl0)), float(np.max(slf)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, sl, t_now, label in [(axes[0], sl0, t_myr[0], "Initial"), (axes[1], slf, t_myr[-1], "Final")]:
        im = ax.imshow(sl.T, origin="lower", extent=extent, cmap="plasma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{label} gas density, t={t_now:.0f} Myr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(out_dir, "gas_density_slices.png"), dpi=140)
    plt.close(fig)


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
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "velocity_history.png"), dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    if args.quick:
        args.n_steps = 24
        args.snapshot_every = 4
        args.max_dt = 15.0
        args.n_grid = min(int(args.n_grid), 32)
        args.n_par_main = min(int(args.n_par_main), 4096)

    out_dir = os.path.join(OUT_ROOT, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    clusters = cluster_run6_setup()
    if args.n_par_3 is None:
        args.n_par_3 = max(512, int(round(args.n_par_main * (M200_3 / M200_1))))
    n_pars = [int(args.n_par_main), int(args.n_par_main), int(args.n_par_3)]

    print("[Triple Merger] Building static-background three-cluster ICs...")
    U0, rho_dm, meta = build_triple_gas_state(clusters, int(args.n_grid), float(args.l_box))
    dm_particles = None
    if args.dm_mode == "particles":
        dm_particles = build_triple_dm_particles(clusters, n_pars, float(args.r_max), float(args.l_box), int(args.seed))

    print(f"  centers={[cl['center'].tolist() for cl in clusters]}")
    print(f"  velocities={[cl['velocity'].tolist() for cl in clusters]}")
    print(f"  gas peak rho={float(np.max(U0[0])):.3e} Msun/kpc^3")
    if dm_particles is not None:
        m = np.asarray(dm_particles['m_par'], dtype=np.float64)
        print(
            f"  live DM: N_total={dm_particles['n_par_total']:,}, "
            f"m_par range=[{float(np.min(m)):.3e}, {float(np.max(m)):.3e}] Msun"
        )

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

    plot_density_slices(U_snaps, t_myr, int(args.n_grid), float(args.l_box), out_dir)
    density_slice_animation(U_snaps, t_myr, int(args.n_grid), float(args.l_box), out_dir)
    s2.make_temperature_animation(U_snaps, t_myr, int(args.n_grid), float(args.l_box), out_dir)
    plot_velocity_history(U_snaps, t_myr, out_dir)
    if dm_particles is not None:
        s2.plot_dm_profile_evolution(dm_pos_snaps, t_myr, dm_particles["m_par"], int(args.n_grid), float(args.l_box))
        s2.make_dm_density_animation(dm_pos_snaps, t_myr, dm_particles["m_par"], int(args.n_grid), float(args.l_box), out_dir)

    scalars = {
        "n_grid": int(args.n_grid),
        "l_box_kpc": float(args.l_box),
        "dx_kpc": float(args.l_box) / int(args.n_grid),
        "n_steps": int(args.n_steps),
        "max_dt_Myr": float(args.max_dt),
        "snapshot_every": int(args.snapshot_every),
        "t_final_Myr": float(t_myr[-1]),
        "t_wall_s": float(t_wall),
        "dm_mode": str(args.dm_mode),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "n_par_1": int(n_pars[0]) if dm_particles is not None else 0,
        "n_par_2": int(n_pars[1]) if dm_particles is not None else 0,
        "n_par_3": int(n_pars[2]) if dm_particles is not None else 0,
        "n_par_total": int(dm_particles["n_par_total"]) if dm_particles is not None else 0,
        "rho_bg_Msun_kpc3": float(meta["rho_bg"]),
        "p_bg_Msun_kpc_Myr2": float(meta["p_bg"]),
        "rho_peak_initial": float(np.max(np.asarray(U_snaps[0][0]))),
        "rho_peak_final": float(np.max(np.asarray(U_snaps[-1][0]))),
    }
    with open(os.path.join(out_dir, "scalars.yaml"), "w") as fh:
        yaml.safe_dump(scalars, fh, sort_keys=False)
    print(f"[Triple Merger] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
