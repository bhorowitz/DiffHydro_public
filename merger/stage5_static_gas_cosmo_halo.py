"""
stage5_static_gas_cosmo_halo.py
================================
Stage 5 diagnostic: gas-only cosmological evolution in a static analytic halo.

This isolates the cosmological hydro update and unit conversions by evolving the
HSE gas profile in the analytic halo gravity field, without live DM coupling.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

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

import jax
import jax.numpy as jnp

import diffhydro as dh
from diffhydro.cosmology import BackgroundExpansionForce, LCDMBackground, SuperComovingEquationManager
from diffhydro.cosmology import conversions as cosmo_conv
from diffhydro.equationmanager import EquationManager

from merger.halo_reference import HaloReference
from merger.halo_gas_ic import build_stage2_gas_grids, shell_profile_from_grid


OUT_DIR = os.path.join(_HERE, "outputs", "stage5_static")
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


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 gas-only static-potential cosmological test")
    p.add_argument("--n-grid", type=int, default=32, help="Mesh resolution")
    p.add_argument("--a-final", type=float, default=0.80, help="Final scale factor")
    p.add_argument("--max-steps", type=int, default=80, help="Maximum number of hydro steps")
    p.add_argument("--snapshot-every", type=int, default=10, help="Snapshot cadence in steps")
    p.add_argument("--a-tol", type=float, default=1.0e-6, help="Stop once a is within this tolerance of a_final")
    p.add_argument("--max-dtau", type=float, default=1.0, help="Maximum supercomoving timestep")
    p.add_argument("--min-dtau", type=float, default=1.0e-3, help="Minimum supercomoving timestep")
    p.add_argument("--dtau-safety", type=float, default=0.8, help="Safety factor on the remaining-a estimate")
    p.add_argument("--pressure-scale", type=float, default=1.0, help="Pressure scale applied to the HSE profile")
    p.add_argument("--cancel-hubble-flow", action="store_true", help="Initialize peculiar velocities to cancel local Hubble expansion for a static physical halo")
    p.add_argument("--gravity-scale", type=float, default=1.0, help="Multiplicative factor applied to the gas gravity kick")
    p.add_argument("--output-subdir", type=str, default=None, help="Optional run subdirectory inside merger/outputs/stage5_static")
    p.add_argument("--l-box", type=float, default=None, help="Initial physical box size [kpc]")
    return p.parse_args()


def hubble_to_myr_inv(h0_km_s_mpc: float) -> float:
    return float(h0_km_s_mpc) * KM_S_TO_KPC_MYR / 1000.0


def temperature_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30)


def entropy_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30) ** GAMMA


def primitives_from_U(eq, U):
    W = np.asarray(eq.get_primitives_from_conservatives(jnp.asarray(U)))
    return {"rho": W[0], "vx": W[1], "vy": W[2], "vz": W[3], "p": W[4]}


class StaticGasGravityForce:
    """Apply static-physical halo gravity to gas in supercomoving variables."""

    def __init__(self, x_com, y_com, z_com, center_com, radius_table, g_table, eps=1.0e-20, gravity_scale=1.0):
        self.x_com = jnp.asarray(x_com, dtype=jnp.float32)
        self.y_com = jnp.asarray(y_com, dtype=jnp.float32)
        self.z_com = jnp.asarray(z_com, dtype=jnp.float32)
        self.center_com = jnp.asarray(center_com, dtype=jnp.float32)
        self.radius_table = jnp.asarray(radius_table, dtype=jnp.float32)
        self.g_table = jnp.asarray(g_table, dtype=jnp.float32)
        self.eps = float(eps)
        self.gravity_scale = float(gravity_scale)

    def timestep(self, U):
        del U
        return jnp.asarray(1.0e10, dtype=jnp.float32)

    def force(self, i, U, params, dtau):
        del i
        dtau = jnp.maximum(jnp.asarray(dtau), 0.0)
        a = jnp.asarray(params.get("a", 1.0), dtype=jnp.float32)
        rho = jnp.asarray(U[0], dtype=jnp.float32)

        dx_com = self.x_com - self.center_com[0]
        dy_com = self.y_com - self.center_com[1]
        dz_com = self.z_com - self.center_com[2]
        r_com = jnp.sqrt(dx_com**2 + dy_com**2 + dz_com**2)
        r_phys = a * r_com
        g_r = jnp.interp(
            r_phys.reshape((-1,)),
            self.radius_table,
            self.g_table,
            left=self.g_table[0],
            right=0.0,
        ).reshape(r_phys.shape)
        inv_r = 1.0 / jnp.maximum(r_com, 1.0e-30)
        gx_phys = g_r * dx_com * inv_r
        gy_phys = g_r * dy_com * inv_r
        gz_phys = g_r * dz_com * inv_r
        gx_phys = jnp.where(r_com > 0.0, gx_phys, 0.0)
        gy_phys = jnp.where(r_com > 0.0, gy_phys, 0.0)
        gz_phys = jnp.where(r_com > 0.0, gz_phys, 0.0)

        ax_code = self.gravity_scale * (a**3) * gx_phys
        ay_code = self.gravity_scale * (a**3) * gy_phys
        az_code = self.gravity_scale * (a**3) * gz_phys

        rho_safe = jnp.maximum(rho, self.eps)
        rho_kin = jnp.maximum(rho_safe, 1.0e-10)
        mx_old, my_old, mz_old, E_old = U[1], U[2], U[3], U[4]

        dmx = rho * ax_code * dtau
        dmy = rho * ay_code * dtau
        dmz = rho * az_code * dtau
        mx_new = mx_old + dmx
        my_new = my_old + dmy
        mz_new = mz_old + dmz

        kin_old = 0.5 * (mx_old * mx_old + my_old * my_old + mz_old * mz_old) / rho_kin
        kin_new = 0.5 * (mx_new * mx_new + my_new * my_new + mz_new * mz_new) / rho_kin
        eint = jnp.maximum(E_old - kin_old, self.eps)
        E_new = jnp.maximum(eint + kin_new, kin_new + self.eps)

        U_new = U.at[1].set(mx_new).at[2].set(my_new).at[3].set(mz_new).at[4].set(E_new)
        return U_new, params


def apply_state_floors(eq, U, a, rho_floor_phys=1.0e-12, p_floor_phys=1.0e-14, t_floor=1.0e-12):
    w = eq.get_primitives_from_conservatives(U)
    w = jnp.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    rho_floor = cosmo_conv.density_phys_to_code(jnp.asarray(rho_floor_phys, dtype=w.dtype), a)
    p_floor = cosmo_conv.pressure_phys_to_code(jnp.asarray(p_floor_phys, dtype=w.dtype), a)
    rho_phys = cosmo_conv.density_code_to_phys(w[0], a)
    p_floor_temp_phys = jnp.maximum(rho_phys, jnp.asarray(rho_floor_phys, dtype=w.dtype)) * jnp.asarray(eq.R * t_floor, dtype=w.dtype)
    p_floor_temp = cosmo_conv.pressure_phys_to_code(p_floor_temp_phys, a)
    p_floor = jnp.maximum(p_floor, p_floor_temp)
    w = w.at[0].set(jnp.maximum(w[0], rho_floor))
    w = w.at[4].set(jnp.maximum(w[4], p_floor))
    return eq.get_conservatives_from_primitives(w)


def build_initial_state(ref, args):
    a_init = 1.0 / (1.0 + Z_INIT)
    l_box_phys = args.l_box if args.l_box else 4.0 * ref.r200
    l_box_com = l_box_phys / a_init

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

    grids = build_stage2_gas_grids(ref, args.n_grid, l_box_phys)
    rho_phys = grids["rho_g"].astype(np.float32)
    p_phys = (grids["p_g"] * float(args.pressure_scale)).astype(np.float32)
    if args.cancel_hubble_flow:
        H_init = float(bg.H(a_init))
        vx_phys = (-H_init * (grids["X"] - grids["center"][0])).astype(np.float32)
        vy_phys = (-H_init * (grids["Y"] - grids["center"][1])).astype(np.float32)
        vz_phys = (-H_init * (grids["Z"] - grids["center"][2])).astype(np.float32)
    else:
        vx_phys = np.zeros_like(rho_phys)
        vy_phys = np.zeros_like(rho_phys)
        vz_phys = np.zeros_like(rho_phys)
    W_code = eq.physical_primitives_to_code(
        jnp.asarray(rho_phys),
        jnp.asarray(vx_phys),
        jnp.asarray(vy_phys),
        jnp.asarray(vz_phys),
        jnp.asarray(p_phys),
        jnp.asarray(a_init, dtype=jnp.float32),
    )
    U0 = eq.get_conservatives_from_primitives(W_code)

    rr = np.maximum(grids["r3d"], 1.0e-30)
    meta = {
        "a_init": float(a_init),
        "l_box_phys": float(l_box_phys),
        "l_box_com": float(l_box_com),
        "dx_phys": float(grids["dx"]),
        "grids": grids,
        "cancel_hubble_flow": bool(args.cancel_hubble_flow),
        "H_init_myr_inv": float(bg.H(a_init)),
    }
    x_com = grids["X"] / a_init
    y_com = grids["Y"] / a_init
    z_com = grids["Z"] / a_init
    center_com = grids["center"] / a_init
    return eq, bg, U0, {"a": jnp.asarray(a_init, dtype=jnp.float32)}, meta, x_com, y_com, z_com, center_com


def snapshot_from_state(eq, U, params, meta):
    a = float(params["a"])
    prim_code = primitives_from_U(eq, U)
    rho_phys = np.asarray(cosmo_conv.density_code_to_phys(jnp.asarray(prim_code["rho"]), a))
    p_phys = np.asarray(cosmo_conv.pressure_code_to_phys(jnp.asarray(prim_code["p"]), a))
    vx_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vx"]), a))
    vy_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vy"]), a))
    vz_phys = np.asarray(cosmo_conv.velocity_code_to_phys(jnp.asarray(prim_code["vz"]), a))
    return {
        "a": a,
        "rho_phys": rho_phys,
        "p_phys": p_phys,
        "T_phys": temperature_proxy(rho_phys, p_phys),
        "entropy_phys": entropy_proxy(rho_phys, p_phys),
        "vx_phys": vx_phys,
        "vy_phys": vy_phys,
        "vz_phys": vz_phys,
    }


def current_physical_grid(meta, a):
    scale = float(a) / float(meta["a_init"])
    grids = meta["grids"]
    dx = float(meta["dx_phys"]) * scale
    x = (grids["X"] - grids["center"][0]) * scale
    y = (grids["Y"] - grids["center"][1]) * scale
    z = (grids["Z"] - grids["center"][2]) * scale
    r = grids["r3d"] * scale
    return dx, x, y, z, r


def choose_dtau(sim, U, bg, a_now, a_final, max_dtau, min_dtau, dtau_safety):
    da_dtau = float(bg.da_dtau(a_now))
    remaining = max(float(a_final) - float(a_now), 0.0)
    dtau_target = min(float(max_dtau), float(dtau_safety) * remaining / max(da_dtau, 1.0e-12))
    dtau_cfl = float(sim.timestep(U))
    dtau = min(dtau_cfl, dtau_target)
    if dtau_target > float(min_dtau):
        dtau = max(dtau, float(min_dtau))
        dtau = min(dtau, dtau_cfl, dtau_target)
    return max(dtau, min(dtau_cfl, dtau_target))


def validate_finite_state(U, params, where):
    U_np = np.asarray(U)
    a_val = float(params["a"])
    if not np.isfinite(U_np).all():
        raise FloatingPointError(f"Non-finite hydro state encountered at {where}.")
    if not np.isfinite(a_val):
        raise FloatingPointError(f"Non-finite scale factor encountered at {where}.")


def compute_snapshot_metrics(ref, snap, meta, pressure_scale):
    grids = meta["grids"]
    prof = grids["profiles"]
    r_prof = prof["radius"]
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * ref.r200), 48)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    dx_phys_now, x_phys, y_phys, z_phys, r_phys_grid = current_physical_grid(meta, snap["a"])

    rho_prof = shell_profile_from_grid(snap["rho_phys"], r_phys_grid, r_bins, statistic="mean")
    p_prof = shell_profile_from_grid(snap["p_phys"], r_phys_grid, r_bins, statistic="mean")
    dPdr = np.gradient(p_prof, r_c, edge_order=1)
    g_shell = np.interp(r_c, r_prof, prof["gravitational_field"])
    resid = (dPdr - rho_prof * g_shell) / (np.abs(dPdr) + np.abs(rho_prof * g_shell) + 1.0e-30)

    vx, vy, vz = snap["vx_phys"], snap["vy_phys"], snap["vz_phys"]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    cs = np.sqrt(GAMMA * snap["p_phys"] / np.maximum(snap["rho_phys"], 1.0e-30))
    mach = vmag / np.maximum(cs, 1.0e-30)
    etherm = float(np.sum(snap["p_phys"] / (GAMMA - 1.0)) * dx_phys_now**3)
    ekin = float(0.5 * np.sum(snap["rho_phys"] * (vx**2 + vy**2 + vz**2)) * dx_phys_now**3)

    target_rho = prof["gas_density"]
    target_p = prof["pressure"] * pressure_scale
    rho_rel = shell_profile_from_grid(snap["rho_phys"], r_phys_grid, r_bins, statistic="mean") / np.maximum(np.interp(r_c, r_prof, target_rho), 1.0e-30)
    p_rel = shell_profile_from_grid(snap["p_phys"], r_phys_grid, r_bins, statistic="mean") / np.maximum(np.interp(r_c, r_prof, target_p), 1.0e-30)

    return {
        "a": float(snap["a"]),
        "hse_residual_rms": float(np.sqrt(np.nanmean(resid**2))),
        "hse_residual_p95": float(np.nanpercentile(np.abs(resid), 95.0)),
        "mach95": float(np.nanpercentile(mach, 95.0)),
        "kinetic_to_thermal": float(ekin / max(etherm, 1.0e-30)),
        "rho_profile_rel_mean": float(np.nanmean(np.abs(np.log(np.maximum(rho_rel, 1.0e-30))))),
        "p_profile_rel_mean": float(np.nanmean(np.abs(np.log(np.maximum(p_rel, 1.0e-30))))),
        "min_temperature": float(np.nanmin(snap["T_phys"])),
        "min_pressure": float(np.nanmin(snap["p_phys"])),
    }


def plot_profiles(ref, snapshots, meta, pressure_scale, out_dir):
    grids = meta["grids"]
    prof = grids["profiles"]
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * prof["radius"][-1]), 60)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    select_idx = np.unique(np.linspace(0, len(snapshots) - 1, min(4, len(snapshots))).astype(int))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    axes[0].loglog(prof["radius"], prof["gas_density"], "k--", lw=2, label="Target")
    axes[1].loglog(prof["radius"], prof["pressure"] * pressure_scale, "k--", lw=2)
    axes[2].loglog(prof["radius"], temperature_proxy(prof["gas_density"], prof["pressure"] * pressure_scale), "k--", lw=2)
    axes[3].loglog(prof["radius"], entropy_proxy(prof["gas_density"], prof["pressure"] * pressure_scale), "k--", lw=2)

    for idx in select_idx:
        snap = snapshots[idx]
        _, x_phys, y_phys, z_phys, r_phys_grid = current_physical_grid(meta, snap["a"])
        rho = shell_profile_from_grid(snap["rho_phys"], r_phys_grid, r_bins, statistic="mean")
        p = shell_profile_from_grid(snap["p_phys"], r_phys_grid, r_bins, statistic="mean")
        t = shell_profile_from_grid(snap["T_phys"], r_phys_grid, r_bins, statistic="mean")
        s = shell_profile_from_grid(snap["entropy_phys"], r_phys_grid, r_bins, statistic="mean")
        rr = np.maximum(r_phys_grid, 1.0e-20)
        vr = (snap["vx_phys"] * x_phys + snap["vy_phys"] * y_phys + snap["vz_phys"] * z_phys) / rr
        vr_prof = shell_profile_from_grid(vr, r_phys_grid, r_bins, statistic="mean")
        cs = np.sqrt(GAMMA * snap["p_phys"] / np.maximum(snap["rho_phys"], 1.0e-30))
        mach_prof = shell_profile_from_grid(np.sqrt(snap["vx_phys"]**2 + snap["vy_phys"]**2 + snap["vz_phys"]**2) / np.maximum(cs, 1.0e-30), r_phys_grid, r_bins, statistic="mean")
        label = f"a={snap['a']:.3f}"
        axes[0].loglog(r_c, rho, label=label)
        axes[1].loglog(r_c, p)
        axes[2].loglog(r_c, t)
        axes[3].loglog(r_c, s)
        axes[4].semilogx(r_c, vr_prof, label=label)
        axes[5].semilogx(r_c, mach_prof, label=label)

    titles = ["Gas density", "Pressure", "Temperature", "Entropy", "Radial velocity", "Mach profile"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("r [kpc]")
    axes[4].axhline(0.0, color="k", lw=1)
    axes[5].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    axes[4].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profiles_multi_a.png"), dpi=140)
    plt.close()


def plot_gas_slices(snapshots, out_dir):
    snap_i = snapshots[0]
    snap_f = snapshots[-1]
    N = snap_i["rho_phys"].shape[0]
    fields_i = [np.log10(snap_i["rho_phys"][:, :, N // 2] + 1.0e-30), np.log10(snap_i["p_phys"][:, :, N // 2] + 1.0e-30), np.log10(snap_i["T_phys"][:, :, N // 2] + 1.0e-30)]
    fields_f = [np.log10(snap_f["rho_phys"][:, :, N // 2] + 1.0e-30), np.log10(snap_f["p_phys"][:, :, N // 2] + 1.0e-30), np.log10(snap_f["T_phys"][:, :, N // 2] + 1.0e-30)]
    titles = [r"$\log_{10}\rho_g$", r"$\log_{10}p$", r"$\log_{10}T$"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j in range(3):
        vmin = min(np.min(fields_i[j]), np.min(fields_f[j]))
        vmax = max(np.max(fields_i[j]), np.max(fields_f[j]))
        im = axes[0, j].imshow(fields_i[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"Initial {titles[j]}")
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
        im2 = axes[1, j].imshow(fields_f[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"Final {titles[j]}")
        plt.colorbar(im2, ax=axes[1, j], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gas_slices_initial_final.png"), dpi=140)
    plt.close()


def make_gas_slice_animation(snapshots, out_dir):
    if len(snapshots) < 2:
        return
    N = snapshots[0]["rho_phys"].shape[0]
    dens_slices = [np.log10(snap["rho_phys"][:, :, N // 2] + 1.0e-30) for snap in snapshots]
    temp_slices = [np.log10(snap["T_phys"][:, :, N // 2] + 1.0e-30) for snap in snapshots]
    dens_vmin = min(np.min(arr) for arr in dens_slices)
    dens_vmax = max(np.max(arr) for arr in dens_slices)
    temp_vmin = min(np.min(arr) for arr in temp_slices)
    temp_vmax = max(np.max(arr) for arr in temp_slices)
    frames = []
    for snap, dens, temp in zip(snapshots, dens_slices, temp_slices):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        im0 = axes[0].imshow(dens, origin="lower", cmap="viridis", vmin=dens_vmin, vmax=dens_vmax)
        axes[0].set_title(rf"Gas density  a={snap['a']:.3f}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(temp, origin="lower", cmap="magma", vmin=temp_vmin, vmax=temp_vmax)
        axes[1].set_title(rf"Temperature  a={snap['a']:.3f}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(os.path.join(out_dir, "gas_slices_evolution.gif"), frames, duration=0.8, loop=0)


def plot_time_series(history, out_dir):
    a = np.array([h["a"] for h in history])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(a, np.array([h["hse_residual_rms"] for h in history]), "o-")
    axes[0, 0].set_title("HSE residual rms")
    axes[0, 1].plot(a, np.array([h["kinetic_to_thermal"] for h in history]), "o-")
    axes[0, 1].set_title("Gas kinetic / thermal")
    axes[0, 2].plot(a, np.array([h["mach95"] for h in history]), "o-")
    axes[0, 2].set_title("Mach95")
    axes[1, 0].plot(a, np.array([h["rho_profile_rel_mean"] for h in history]), "o-")
    axes[1, 0].set_title("Density-profile drift")
    axes[1, 1].plot(a, np.array([h["p_profile_rel_mean"] for h in history]), "o-")
    axes[1, 1].set_title("Pressure-profile drift")
    axes[1, 2].plot(a, np.array([h["min_temperature"] for h in history]), "o-")
    axes[1, 2].set_title("Minimum temperature")
    for ax in axes.ravel():
        ax.set_xlabel("a")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_series_metrics.png"), dpi=140)
    plt.close()


def write_history_bundle(history, out_dir):
    np.savez(
        os.path.join(out_dir, "snapshot_history.npz"),
        a=np.array([h["a"] for h in history], dtype=np.float32),
        hse_residual_rms=np.array([h["hse_residual_rms"] for h in history], dtype=np.float32),
        kinetic_to_thermal=np.array([h["kinetic_to_thermal"] for h in history], dtype=np.float32),
        mach95=np.array([h["mach95"] for h in history], dtype=np.float32),
        rho_profile_rel_mean=np.array([h["rho_profile_rel_mean"] for h in history], dtype=np.float32),
        p_profile_rel_mean=np.array([h["p_profile_rel_mean"] for h in history], dtype=np.float32),
    )


def main():
    global OUT_DIR
    args = parse_args()
    if args.output_subdir:
        OUT_DIR = os.path.join(_HERE, "outputs", "stage5_static", args.output_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = HaloReference(z=Z_INIT, m200_msun=M200, conc=CONC)
    eq, bg, U, params, meta, x_com, y_com, z_com, center_com = build_initial_state(ref, args)
    solver = dh.HLLC(equation_manager=eq, signal_speed=dh.signal_speed_Rusanov)
    flux = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="VANLEER"), positivity=True)
    flux.dx_o = meta["l_box_com"] / args.n_grid
    sim = dh.hydro(
        n_super_step=1,
        max_dt=0.2,
        fluxes=[flux],
        forces=[
            BackgroundExpansionForce(bg, a_init=meta["a_init"]),
            StaticGasGravityForce(
                x_com,
                y_com,
                z_com,
                center_com,
                meta["grids"]["profiles"]["radius"],
                meta["grids"]["profiles"]["gravitational_field"],
                gravity_scale=args.gravity_scale,
            ),
        ],
        use_mol=True,
        pmesh_shape=(1, 1, 1),
        dx_o=float(flux.dx_o),
    )

    snapshots = []
    history = []
    snapshots.append(snapshot_from_state(eq, U, params, meta))
    history.append(compute_snapshot_metrics(ref, snapshots[-1], meta, args.pressure_scale))

    a_init = meta["a_init"]
    a_final = max(float(args.a_final), a_init + 1.0e-5)
    wall_t0 = time.time()
    step_i = 0
    while (a_final - float(params["a"])) > float(args.a_tol):
        if step_i >= args.max_steps:
            raise RuntimeError(f"Reached max_steps={args.max_steps} before a_final={a_final:.4f}.")
        a_now = float(params["a"])
        dtau = choose_dtau(sim, U, bg, a_now, a_final, args.max_dtau, args.min_dtau, args.dtau_safety)
        U, params = sim._hydrostep(step_i, (U, params), jnp.asarray(dtau, dtype=jnp.float32))
        U = apply_state_floors(eq, U, jnp.asarray(params["a"], dtype=jnp.float32))
        validate_finite_state(U, params, f"step {step_i}")
        step_i += 1
        if (step_i % args.snapshot_every) == 0 or float(params["a"]) >= a_final:
            snapshots.append(snapshot_from_state(eq, U, params, meta))
            history.append(compute_snapshot_metrics(ref, snapshots[-1], meta, args.pressure_scale))

    wall = time.time() - wall_t0
    plot_profiles(ref, snapshots, meta, args.pressure_scale, OUT_DIR)
    plot_gas_slices(snapshots, OUT_DIR)
    make_gas_slice_animation(snapshots, OUT_DIR)
    plot_time_series(history, OUT_DIR)
    write_history_bundle(history, OUT_DIR)

    scalars = {
        "a_init": float(a_init),
        "a_final": float(history[-1]["a"]),
        "pressure_scale": float(args.pressure_scale),
        "gravity_scale": float(args.gravity_scale),
        "cancel_hubble_flow": bool(args.cancel_hubble_flow),
        "steps_taken": int(step_i),
        "snapshot_count": int(len(snapshots)),
        "wall_time_s": float(wall),
        "hse_residual_rms_init": float(history[0]["hse_residual_rms"]),
        "hse_residual_rms_final": float(history[-1]["hse_residual_rms"]),
        "mach95_final": float(history[-1]["mach95"]),
        "rho_profile_rel_mean_final": float(history[-1]["rho_profile_rel_mean"]),
        "p_profile_rel_mean_final": float(history[-1]["p_profile_rel_mean"]),
        "min_temperature_final": float(history[-1]["min_temperature"]),
        "min_pressure_final": float(history[-1]["min_pressure"]),
    }
    params_out = {
        "stage": "5-static",
        "description": "Gas-only cosmological evolution in static analytic halo gravity",
        "halo": {
            "z_init": float(Z_INIT),
            "M200_Msun": float(M200),
            "conc": float(CONC),
            "r200_kpc": float(ref.r200),
            "r500_kpc": float(ref.r500),
        },
        "cosmology": {
            "H0_km_s_Mpc": float(H0_KM_S_MPC),
            "H0_Myr_inv": float(bg.H0),
            "Omega_m": float(OMEGA_M),
            "a_init": float(a_init),
            "a_final_target": float(a_final),
        },
        "run": {
            "n_grid": int(args.n_grid),
            "l_box_phys_kpc": float(meta["l_box_phys"]),
            "l_box_com_kpc": float(meta["l_box_com"]),
            "dx_phys_kpc": float(meta["dx_phys"]),
            "dx_com_kpc": float(meta["l_box_com"] / args.n_grid),
            "snapshot_every": int(args.snapshot_every),
            "max_steps": int(args.max_steps),
            "max_dtau": float(args.max_dtau),
            "min_dtau": float(args.min_dtau),
            "dtau_safety": float(args.dtau_safety),
            "a_tol": float(args.a_tol),
            "pressure_scale": float(args.pressure_scale),
            "gravity_scale": float(args.gravity_scale),
            "cancel_hubble_flow": bool(args.cancel_hubble_flow),
        },
    }

    with open(os.path.join(OUT_DIR, "stage5_static_params.yaml"), "w") as f:
        yaml.dump(params_out, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(OUT_DIR, "scalars.txt"), "w") as f:
        f.write("Stage 5 Static Scalar Diagnostics\n")
        for k, v in scalars.items():
            f.write(f"{k}: {v}\n")

    print("[Stage 5 Static] Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
