"""
stage2_gas_dm_halo_perturb.py
=============================
Stage 2 single-cluster run with a small off-center gas perturbation.

This script reuses the non-cosmological, static-DM + live gas-gravity setup
from ``stage2_gas_dm_halo.py`` and adds a Gaussian gas overdensity intended to
act like a minor merger / infalling gas clump.

Outputs are written to ``merger/outputs/stage2_perturb/<subdir>`` and include
an animated GIF of the gas density evolution in the midplane.
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
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import merger.stage2_gas_dm_halo as s2


OUT_ROOT = os.path.join(_HERE, "outputs", "stage2_perturb")
os.makedirs(OUT_ROOT, exist_ok=True)
COSMIC_DM_TO_BARYON_RATIO = (0.3 - 0.0486) / 0.0486


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 gas+DM single-cluster run with an off-center gas perturbation")
    p.add_argument("--n-steps", type=int, default=160, help="Number of evolution steps")
    p.add_argument("--max-dt", type=float, default=25.0, help="Maximum timestep in Myr")
    p.add_argument("--snapshot-every", type=int, default=8, help="Snapshot cadence in steps")
    p.add_argument("--dm-mode", type=str, default="particles", choices=["particles", "static"], help="Use live DM particles or the old static analytic DM field")
    p.add_argument("--n-par", type=int, default=64**3, help="Number of live DM particles when --dm-mode=particles")
    p.add_argument("--r-max", type=float, default=None, help="DM particle truncation radius [kpc] when --dm-mode=particles")
    p.add_argument("--gravity-method", type=str, default="pm", choices=["pm", "direct"], help="DM particle gravity method when --dm-mode=particles")
    p.add_argument("--softening-cells", type=float, default=0.5, help="Plummer softening for direct live-DM particle forces, in cell units")
    p.add_argument("--perturb-radius", type=float, default=2000.0, help="Perturbation center radius from halo center [kpc]")
    p.add_argument("--perturb-width", type=float, default=220.0, help="Gaussian sigma of perturbation [kpc]")
    p.add_argument("--perturb-rho-peak", type=float, default=1.0e4, help="Peak added gas density [Msun/kpc^3]")
    p.add_argument("--perturb-axis", type=str, default="x", choices=["x", "y", "z"], help="Axis along which the perturbation is offset")
    p.add_argument("--no-companion-dm", action="store_true", help="Disable the DM halo attached to the perturbation")
    p.add_argument("--companion-dm-ratio", type=float, default=COSMIC_DM_TO_BARYON_RATIO, help="Target companion DM-to-gas mass ratio")
    p.add_argument("--companion-n-par", type=int, default=None, help="Override the number of companion DM particles")
    p.add_argument("--companion-r-max", type=float, default=None, help="Companion DM halo truncation radius [kpc]; default 3*sigma")
    p.add_argument("--companion-seed", type=int, default=1234, help="Random seed for the companion DM subset sampling")
    p.add_argument("--output-subdir", type=str, default="minor_merger_r2mpc_rho1e4", help="Output subdirectory inside merger/outputs/stage2_perturb")
    p.add_argument("--quick", action="store_true", help="Quick smoke test")
    return p.parse_args()


def gaussian_perturbation(n_grid, l_box, *, radius_kpc, sigma_kpc, peak_rho, axis="x"):
    dx = l_box / n_grid
    x = (np.arange(n_grid) + 0.5) * dx
    cen = l_box / 2.0
    X, Y, Z = np.meshgrid(x - cen, x - cen, x - cen, indexing="ij")

    center = {"x": np.array([radius_kpc, 0.0, 0.0], dtype=np.float64),
              "y": np.array([0.0, radius_kpc, 0.0], dtype=np.float64),
              "z": np.array([0.0, 0.0, radius_kpc], dtype=np.float64)}[axis]
    rr = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    rho_add = peak_rho * np.exp(-0.5 * (rr / max(sigma_kpc, 1.0e-12))**2)
    return rho_add.astype(np.float32), center.astype(np.float32)


def gaussian_density_field(n_grid, l_box, *, center_offset, sigma_kpc, total_mass):
    dx = l_box / n_grid
    x = (np.arange(n_grid) + 0.5) * dx
    cen = l_box / 2.0
    X, Y, Z = np.meshgrid(x - cen, x - cen, x - cen, indexing="ij")
    rr = np.sqrt((X - center_offset[0])**2 + (Y - center_offset[1])**2 + (Z - center_offset[2])**2)
    norm = total_mass / max((2.0 * np.pi) ** 1.5 * sigma_kpc**3, 1.0e-30)
    rho = norm * np.exp(-0.5 * (rr / max(sigma_kpc, 1.0e-12))**2)
    return rho.astype(np.float32)


def add_isothermal_density_perturbation(U0, rho_add):
    """Add a density perturbation while keeping the local temperature fixed."""
    U = np.array(U0, copy=True)
    rho0 = np.maximum(U0[0], 1.0e-30)
    p0 = U0[4] * (s2.GAMMA - 1.0)
    temperature_like = p0 / rho0

    rho_new = rho0 + rho_add
    p_new = temperature_like * rho_new

    U[0] = rho_new.astype(np.float32)
    U[4] = (p_new / (s2.GAMMA - 1.0)).astype(np.float32)
    return U


def build_companion_dm_particles(template_dm, l_box, center_offset, *, target_dm_mass, target_r_max, seed):
    rng = np.random.default_rng(seed)
    pos_centered = np.asarray(template_dm["pos_centered"], dtype=np.float32)
    vel_centered = np.asarray(template_dm["vel"], dtype=np.float32)
    m_par = float(template_dm["m_par"])

    n_template = pos_centered.shape[0]
    n_comp = max(1, int(round(target_dm_mass / max(m_par, 1.0e-30))))
    replace = n_comp > n_template
    pick = rng.choice(n_template, size=n_comp, replace=replace)

    pos_sub = np.array(pos_centered[pick], copy=True)
    vel_sub = np.array(vel_centered[pick], copy=True)
    r_sub = np.sqrt(np.sum(pos_sub**2, axis=1))
    r_scale = float(target_r_max) / max(float(np.max(r_sub)), 1.0e-12)
    mass_scale = n_comp / max(float(n_template), 1.0)
    v_scale = np.sqrt(max(mass_scale / max(r_scale, 1.0e-12), 1.0e-12))

    pos_scaled = pos_sub * np.float32(r_scale)
    vel_scaled = vel_sub * np.float32(v_scale)
    center_box = np.array([l_box / 2.0] * 3, dtype=np.float32) + np.asarray(center_offset, dtype=np.float32)
    pos_box = np.mod(pos_scaled + center_box[None, :], l_box).astype(np.float32)
    return {
        "pos": pos_box,
        "vel": vel_scaled.astype(np.float32),
        "m_par": np.float32(m_par),
        "n_par": int(n_comp),
        "r_scale": float(r_scale),
        "v_scale": float(v_scale),
        "target_mass": float(n_comp * m_par),
        "target_r_max": float(target_r_max),
        "sampled_with_replacement": bool(replace),
    }


def merge_dm_particle_states(primary_dm, companion_dm):
    if companion_dm is None:
        return primary_dm
    return {
        "pos": np.concatenate([np.asarray(primary_dm["pos"], dtype=np.float32), np.asarray(companion_dm["pos"], dtype=np.float32)], axis=0),
        "vel": np.concatenate([np.asarray(primary_dm["vel"], dtype=np.float32), np.asarray(companion_dm["vel"], dtype=np.float32)], axis=0),
        "m_par": np.float32(primary_dm["m_par"]),
        "pos_centered": np.asarray(primary_dm["pos_centered"], dtype=np.float32),
        "ref": primary_dm["ref"],
    }


def shell_profile(field_3d, r3d, bins):
    vals = np.zeros(len(bins) - 1, dtype=np.float64)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    for i in range(len(vals)):
        mask = (r3d >= bins[i]) & (r3d < bins[i + 1])
        if np.any(mask):
            vals[i] = float(np.mean(field_3d[mask]))
    return r_mid, vals


def make_density_animation(U_snaps, t_myr, n_grid, l_box, out_dir):
    L = l_box / 2.0
    extent = [-L, L, -L, L]
    rho_slices = [np.asarray(U[0])[:, :, n_grid // 2] for U in U_snaps]
    log_slices = [np.log10(np.maximum(sl, 1.0e-30)) for sl in rho_slices]
    vmin = min(float(np.min(sl)) for sl in log_slices)
    vmax = max(float(np.max(sl)) for sl in log_slices)
    frames = []

    for t_now, rho_sl, log_sl in zip(t_myr, rho_slices, log_slices):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        im0 = axes[0].imshow(
            log_sl.T,
            origin="lower",
            extent=extent,
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[0].set_title(rf"$\log_{{10}}\rho_g$ at $t={t_now:.0f}$ Myr")
        axes[0].set_xlabel("x [kpc]")
        axes[0].set_ylabel("y [kpc]")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        frac = (rho_sl - rho_slices[0]) / np.maximum(rho_slices[0], 1.0e-30)
        lim = max(0.05, float(np.nanpercentile(np.abs(frac), 99.0)))
        im1 = axes[1].imshow(
            frac.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            aspect="equal",
        )
        axes[1].set_title(r"Fractional density change")
        axes[1].set_xlabel("x [kpc]")
        axes[1].set_ylabel("y [kpc]")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    gif_path = os.path.join(out_dir, "density_evolution.gif")
    imageio.mimsave(gif_path, frames, duration=0.6, loop=0)
    print(f"  Saved {gif_path}")


def plot_profiles(U_snaps, t_myr, n_grid, l_box, out_dir):
    dx = l_box / n_grid
    cen = n_grid / 2.0
    ix, iy, iz = np.indices((n_grid, n_grid, n_grid)) + 0.5
    r3d = np.sqrt((ix - cen)**2 + (iy - cen)**2 + (iz - cen)**2) * dx
    bins = np.linspace(0.0, r3d.max(), 80)
    pick = np.unique(np.linspace(0, len(U_snaps) - 1, min(5, len(U_snaps))).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx in pick:
        rho = np.asarray(U_snaps[idx][0])
        pres = np.asarray(U_snaps[idx][4]) * (s2.GAMMA - 1.0)
        r_mid, rho_prof = shell_profile(rho, r3d, bins)
        _, p_prof = shell_profile(pres, r3d, bins)
        label = f"t={t_myr[idx]:.0f} Myr"
        mask_rho = rho_prof > 0
        mask_p = p_prof > 0
        axes[0].loglog(r_mid[mask_rho], rho_prof[mask_rho], lw=2, label=label)
        axes[1].loglog(r_mid[mask_p], p_prof[mask_p], lw=2, label=label)

    axes[0].set_title("Gas density profile")
    axes[1].set_title("Gas pressure profile")
    for ax, ylabel in zip(axes, ["rho [Msun/kpc^3]", "P [Msun kpc^-1 Myr^-2]"]):
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel(ylabel)
        ax.grid(ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(out_dir, "profiles_evolution.png")
    plt.savefig(out, dpi=140)
    plt.close()
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
        p = (s2.GAMMA - 1.0) * (np.asarray(U[4]) - kinetic)
        p = np.maximum(p, 1.0e-30)
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
    out = os.path.join(out_dir, "velocity_history.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


def main():
    args = parse_args()
    if args.quick:
        args.n_steps = 24
        args.snapshot_every = 4
        args.max_dt = 20.0
        if args.dm_mode == "particles":
            args.n_par = min(int(args.n_par), 4096)

    out_dir = os.path.join(OUT_ROOT, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("[Stage 2 Perturb] Building equilibrium state...")
    U0 = s2.build_gas_state(s2.PROFILE_H5, s2.N_GRID, s2.L_BOX)
    rho_dm = s2.build_dm_density_field(s2.PROFILE_H5, s2.N_GRID, s2.L_BOX)

    print("[Stage 2 Perturb] Adding Gaussian gas overdensity...")
    rho_add, center = gaussian_perturbation(
        s2.N_GRID,
        s2.L_BOX,
        radius_kpc=float(args.perturb_radius),
        sigma_kpc=float(args.perturb_width),
        peak_rho=float(args.perturb_rho_peak),
        axis=str(args.perturb_axis),
    )
    U_init = add_isothermal_density_perturbation(U0, rho_add)

    m_add = float(np.sum(rho_add)) * (s2.L_BOX / s2.N_GRID) ** 3
    companion_dm_mass = 0.0 if args.no_companion_dm else float(args.companion_dm_ratio) * m_add
    companion_r_max = float(args.companion_r_max) if args.companion_r_max is not None else 3.0 * float(args.perturb_width)
    companion_dm = None
    dm_particles = None
    if args.dm_mode == "particles":
        dm_r_max = float(args.r_max) if args.r_max is not None else 2.5 * s2.R200
        dm_particles = s2.build_dm_particle_state(int(args.n_par), dm_r_max, s2.L_BOX)
        if companion_dm_mass > 0.0:
            if args.companion_n_par is not None:
                companion_dm_mass = float(args.companion_n_par) * float(dm_particles["m_par"])
            companion_dm = build_companion_dm_particles(
                dm_particles,
                s2.L_BOX,
                center,
                target_dm_mass=companion_dm_mass,
                target_r_max=companion_r_max,
                seed=int(args.companion_seed),
            )
            dm_particles = merge_dm_particle_states(dm_particles, companion_dm)
    elif companion_dm_mass > 0.0:
        rho_dm = rho_dm + gaussian_density_field(
            s2.N_GRID,
            s2.L_BOX,
            center_offset=center,
            sigma_kpc=max(companion_r_max / 3.0, s2.L_BOX / s2.N_GRID),
            total_mass=companion_dm_mass,
        )

    print(
        f"  center_offset={center} kpc, sigma={args.perturb_width:.1f} kpc, "
        f"rho_peak_add={args.perturb_rho_peak:.3e} Msun/kpc^3, "
        f"m_add={m_add:.3e} Msun"
    )
    if companion_dm_mass > 0.0:
        print(f"  companion DM target mass={companion_dm_mass:.3e} Msun, r_max={companion_r_max:.1f} kpc")
    if dm_particles is not None:
        print(f"  live DM: N={int(np.asarray(dm_particles['pos']).shape[0]):,}, m_par={float(dm_particles['m_par']):.3e} Msun")
        if companion_dm is not None:
            print(
                f"  companion live DM: N={int(companion_dm['n_par']):,}, "
                f"mass={float(companion_dm['target_mass']):.3e} Msun, "
                f"r_scale={float(companion_dm['r_scale']):.3f}, "
                f"v_scale={float(companion_dm['v_scale']):.3f}"
            )

    print("[Stage 2 Perturb] Evolving on GPU...")
    wall_t0 = time.time()
    U_snaps, t_myr, t_wall, dm_pos_snaps = s2.run_evolution(
        U_init,
        rho_dm,
        s2.N_GRID,
        s2.L_BOX,
        int(args.n_steps),
        int(args.snapshot_every),
        float(args.max_dt),
        dm_particles=dm_particles,
        gravity_method=args.gravity_method,
        softening_cells=args.softening_cells,
    )
    print(f"[Stage 2 Perturb] Done. Wall time={t_wall:.1f} s, total elapsed={time.time()-wall_t0:.1f} s")

    print("[Stage 2 Perturb] Writing diagnostics...")
    s2.OUT_DIR = out_dir
    s2.plot_density_slices(U_snaps, t_myr, s2.N_GRID, s2.L_BOX)
    make_density_animation(U_snaps, t_myr, s2.N_GRID, s2.L_BOX, out_dir)
    s2.make_temperature_animation(U_snaps, t_myr, s2.N_GRID, s2.L_BOX, out_dir)
    plot_profiles(U_snaps, t_myr, s2.N_GRID, s2.L_BOX, out_dir)
    plot_velocity_history(U_snaps, t_myr, out_dir)
    if dm_particles is not None:
        s2.plot_dm_profile_evolution(dm_pos_snaps, t_myr, dm_particles["m_par"], s2.N_GRID, s2.L_BOX)
        s2.make_dm_density_animation(dm_pos_snaps, t_myr, dm_particles["m_par"], s2.N_GRID, s2.L_BOX, out_dir)

    rho0 = np.asarray(U_snaps[0][0])
    rhof = np.asarray(U_snaps[-1][0])
    scalars = {
        "n_grid": int(s2.N_GRID),
        "l_box_kpc": float(s2.L_BOX),
        "dx_kpc": float(s2.L_BOX / s2.N_GRID),
        "n_steps": int(args.n_steps),
        "max_dt_Myr": float(args.max_dt),
        "snapshot_every": int(args.snapshot_every),
        "t_final_Myr_est": float(t_myr[-1]),
        "t_wall_s": float(t_wall),
        "dm_mode": str(args.dm_mode),
        "gravity_method": str(args.gravity_method),
        "softening_cells": float(args.softening_cells),
        "n_par": int(np.asarray(dm_particles["pos"]).shape[0]) if dm_particles is not None else 0,
        "m_par_Msun": float(dm_particles["m_par"]) if dm_particles is not None else 0.0,
        "perturb_axis": str(args.perturb_axis),
        "perturb_radius_kpc": float(args.perturb_radius),
        "perturb_width_kpc": float(args.perturb_width),
        "perturb_rho_peak_add": float(args.perturb_rho_peak),
        "perturb_mass_Msun": float(m_add),
        "companion_dm_enabled": bool(not args.no_companion_dm),
        "companion_dm_ratio": float(args.companion_dm_ratio),
        "companion_dm_mass_Msun": float(companion_dm_mass),
        "companion_dm_rmax_kpc": float(companion_r_max),
        "companion_dm_n_par": int(companion_dm["n_par"]) if companion_dm is not None else 0,
        "rho_peak_initial": float(np.max(rho0)),
        "rho_peak_final": float(np.max(rhof)),
    }
    scal_path = os.path.join(out_dir, "scalars.yaml")
    with open(scal_path, "w") as fh:
        yaml.safe_dump(scalars, fh, sort_keys=False)
    print(f"  Saved {scal_path}")


if __name__ == "__main__":
    main()
