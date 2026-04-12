"""
stage3_comoving_single_halo.py
==============================
Stage 3: Convert the single physical halo to comoving/supercomoving variables
and validate the round-trip back to physical space.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax.numpy as jnp

from diffhydro.equationmanager import EquationManager
from diffhydro.cosmology.equation_manager import SuperComovingEquationManager
from diffhydro.cosmology import conversions as cosmo_conv

from merger.halo_reference import HaloReference
from merger.halo_gas_ic import build_stage2_gas_grids, shell_profile_from_grid
from merger.physical_pm_force import nfw_enclosed_mass, nfw_acceleration, nfw_circular_velocity, G_PHYS


OUT_DIR = os.path.join(_HERE, "outputs", "stage3")
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA = 5.0 / 3.0
Z = 0.295
M200 = 5.0e14
CONC = 3.5


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 comoving conversion validation")
    p.add_argument("--n-par", type=int, default=4096, help="Number of DM particles")
    p.add_argument("--n-grid", type=int, default=32, help="Grid resolution")
    p.add_argument("--pressure-scale", type=float, default=0.01, help="Scale factor applied to the physical pressure profile before conversion")
    p.add_argument("--r-max", type=float, default=None, help="Particle truncation radius [kpc]")
    p.add_argument("--l-box", type=float, default=None, help="Physical box size [kpc]")
    return p.parse_args()


def entropy_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30) ** GAMMA


def temperature_proxy(rho, p):
    return p / np.maximum(rho, 1.0e-30)


def build_physical_state(ref, n_grid, l_box, n_par, r_max, pressure_scale=0.01):
    grids = build_stage2_gas_grids(ref, n_grid, l_box)
    grids["pressure_scale"] = float(pressure_scale)
    grids["p_g_1d"] = grids["profiles"]["pressure"] * float(pressure_scale)
    grids["p_g"] = grids["p_g"] * float(pressure_scale)
    base_eq = EquationManager()
    base_eq.gamma = GAMMA
    base_eq.mesh_shape = [n_grid, n_grid, n_grid]
    base_eq.box_size = (l_box, l_box, l_box)
    sc_eq = SuperComovingEquationManager(base_eq, enforce_gamma_53=True)

    rho = grids["rho_g"].astype(np.float32)
    p = grids["p_g"].astype(np.float32)
    zeros = np.zeros_like(rho)
    W_phys = np.stack([rho, zeros, zeros, zeros, p], axis=0)

    np.random.seed(42)
    particles = ref.hse.generate_dm_particles(n_par, r_max=r_max)
    pos = np.array(particles[("dm", "particle_position")], dtype=np.float64)
    vel = np.array(particles[("dm", "particle_velocity")], dtype=np.float64)
    m_par = float(np.array(particles[("dm", "particle_mass")])[0])
    return grids, sc_eq, W_phys, pos, vel, m_par


def convert_roundtrip(sc_eq, W_phys, pos_phys, vel_phys, a):
    W_code = sc_eq.physical_primitives_to_code(
        jnp.asarray(W_phys[0]),
        jnp.asarray(W_phys[1]),
        jnp.asarray(W_phys[2]),
        jnp.asarray(W_phys[3]),
        jnp.asarray(W_phys[4]),
        a,
    )
    U_code = sc_eq.get_conservatives_from_primitives(W_code)
    rho_back, vx_back, vy_back, vz_back, p_back = sc_eq.conservatives_to_physical_primitives(U_code, a)
    W_back = np.stack([
        np.asarray(rho_back), np.asarray(vx_back), np.asarray(vy_back),
        np.asarray(vz_back), np.asarray(p_back)
    ], axis=0)

    pos_com = pos_phys / a
    vel_code = a * vel_phys
    pos_back = pos_com * a
    vel_back = vel_code / a

    return np.asarray(W_code), np.asarray(U_code), W_back, pos_com, vel_code, pos_back, vel_back


def plot_roundtrip_profiles(ref, grids, W_phys, W_back, pos_phys, pos_back, m_par, a):
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * grids["profiles"]["radius"][-1]), 60)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])

    rho_phys = shell_profile_from_grid(W_phys[0], grids["r3d"], r_bins, statistic="mean")
    p_phys = shell_profile_from_grid(W_phys[4], grids["r3d"], r_bins, statistic="mean")
    t_phys = shell_profile_from_grid(temperature_proxy(W_phys[0], W_phys[4]), grids["r3d"], r_bins, statistic="mean")

    rho_back = shell_profile_from_grid(W_back[0], grids["r3d"], r_bins, statistic="mean")
    p_back = shell_profile_from_grid(W_back[4], grids["r3d"], r_bins, statistic="mean")
    t_back = shell_profile_from_grid(temperature_proxy(W_back[0], W_back[4]), grids["r3d"], r_bins, statistic="mean")

    dm_phys = np.array([m_par * np.sum(np.sqrt(np.sum(pos_phys**2, axis=1)) < rr) for rr in r_c])
    dm_back = np.array([m_par * np.sum(np.sqrt(np.sum(pos_back**2, axis=1)) < rr) for rr in r_c])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].loglog(r_c, rho_phys, label="Physical")
    axes[0, 0].loglog(r_c, rho_back, "--", label="Round-trip")
    axes[0, 0].set_title("Gas density")
    axes[0, 0].legend()

    axes[0, 1].loglog(r_c, p_phys, label="Physical")
    axes[0, 1].loglog(r_c, p_back, "--", label="Round-trip")
    axes[0, 1].set_title("Pressure")

    axes[1, 0].loglog(r_c, t_phys, label="Physical")
    axes[1, 0].loglog(r_c, t_back, "--", label="Round-trip")
    axes[1, 0].set_title("Temperature proxy")

    axes[1, 1].loglog(r_c, dm_phys, label="Physical DM")
    axes[1, 1].loglog(r_c, dm_back, "--", label="Round-trip DM")
    axes[1, 1].set_title("DM enclosed mass")

    for ax in axes.ravel():
        ax.set_xlabel("r [kpc]")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "roundtrip_profiles.png")
    plt.savefig(out, dpi=140)
    plt.close()


def plot_comoving_histograms(W_code, pos_com, grids, a):
    rho_code = W_code[0]
    p_code = W_code[4]
    mean_rho = float(np.mean(rho_code))
    delta = rho_code / max(mean_rho, 1.0e-30) - 1.0
    vpec_mag = np.sqrt(np.sum((W_code[1:4] / max(a, 1.0e-30)) ** 2, axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(np.log10(rho_code.ravel() + 1.0e-30), bins=60, color="tab:blue")
    axes[0].set_title(r"$\log_{10}\tilde{\rho}_g$")
    axes[1].hist(delta.ravel(), bins=60, color="tab:orange")
    axes[1].set_title(r"$\delta_g = \tilde{\rho}/\langle\tilde{\rho}\rangle - 1$")
    axes[2].hist(np.log10(p_code.ravel() + 1.0e-30), bins=60, color="tab:green")
    axes[2].set_title(r"$\log_{10}\tilde{p}$")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "comoving_histograms.png")
    plt.savefig(out, dpi=140)
    plt.close()


def plot_velocity_and_force_profiles(ref, pos_phys, vel_phys, a):
    r = np.sqrt(np.sum(pos_phys**2, axis=1))
    vr = np.sum(pos_phys * vel_phys, axis=1) / np.maximum(r, 1.0e-30)
    r_bins = np.logspace(np.log10(0.03 * ref.r200), np.log10(1.2 * ref.r200), 50)
    r_c = 0.5 * (r_bins[:-1] + r_bins[1:])
    vr_prof = np.full(len(r_bins) - 1, np.nan)
    vmag_prof = np.full(len(r_bins) - 1, np.nan)
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            vr_prof[i] = np.mean(vr[mask])
            vmag_prof[i] = np.mean(np.sqrt(np.sum(vel_phys[mask] ** 2, axis=1)))

    x_com = r_c / a
    M_enc = nfw_enclosed_mass(r_c, ref.rho_s, ref.a_scale)
    a_phys = nfw_acceleration(r_c, ref.rho_s, ref.a_scale, G=G_PHYS)
    vc_phys = nfw_circular_velocity(r_c, ref.rho_s, ref.a_scale, G=G_PHYS)
    a_shape_com = a_phys * a**2
    vc_code = a * vc_phys

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogx(r_c / ref.r200, vr_prof, label=r"$v_r$")
    axes[0].semilogx(r_c / ref.r200, vmag_prof, label=r"$|v|$")
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].set_title("Physical peculiar velocities")
    axes[0].set_xlabel("r / r200")
    axes[0].set_ylabel("kpc/Myr")
    axes[0].legend()

    axes[1].loglog(x_com, a_shape_com, label=r"$a_{\rm shape,com}$")
    axes[1].loglog(x_com, vc_code, label=r"$\tilde{v}_c$")
    axes[1].set_title("Comoving/supercomoving profile shapes")
    axes[1].set_xlabel("x_com [kpc]")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "velocity_force_profiles.png")
    plt.savefig(out, dpi=140)
    plt.close()


def main():
    args = parse_args()
    ref = HaloReference(z=Z, m200_msun=M200, conc=CONC)
    a = 1.0 / (1.0 + Z)
    r_max = args.r_max if args.r_max else 2.5 * ref.r200
    l_box = args.l_box if args.l_box else 4.0 * ref.r200

    grids, sc_eq, W_phys, pos_phys, vel_phys, m_par = build_physical_state(
        ref, args.n_grid, l_box, args.n_par, r_max, pressure_scale=args.pressure_scale
    )
    W_code, U_code, W_back, pos_com, vel_code, pos_back, vel_back = convert_roundtrip(
        sc_eq, W_phys, pos_phys, vel_phys, a
    )

    plot_roundtrip_profiles(ref, grids, W_phys, W_back, pos_phys, pos_back, m_par, a)
    plot_comoving_histograms(W_code, pos_com, grids, a)
    plot_velocity_and_force_profiles(ref, pos_phys, vel_phys, a)

    rho_err = np.max(np.abs(W_back[0] - W_phys[0]) / np.maximum(W_phys[0], 1.0e-30))
    p_err = np.max(np.abs(W_back[4] - W_phys[4]) / np.maximum(W_phys[4], 1.0e-30))
    v_err = np.max(np.abs(vel_back - vel_phys) / np.maximum(np.abs(vel_phys), 1.0e-8))
    x_err = np.max(np.abs(pos_back - pos_phys) / np.maximum(np.abs(pos_phys), 1.0))

    r200_back = ref.r200
    r500_back = ref.r500
    scalars = {
        "a_init": float(a),
        "rho_roundtrip_max_rel_err": float(rho_err),
        "pressure_roundtrip_max_rel_err": float(p_err),
        "velocity_roundtrip_max_rel_err": float(v_err),
        "position_roundtrip_max_rel_err": float(x_err),
        "r200_rel_err": float((r200_back - ref.r200) / ref.r200),
        "r500_rel_err": float((r500_back - ref.r500) / ref.r500),
        "box_mean_rho_code": float(np.mean(W_code[0])),
        "box_mean_delta_gas": float(np.mean(W_code[0] / np.maximum(np.mean(W_code[0]), 1.0e-30) - 1.0)),
        "dm_speed_code_mean": float(np.mean(np.sqrt(np.sum(vel_code**2, axis=1)))),
        "pressure_scale": float(args.pressure_scale),
    }
    params = {
        "stage": 3,
        "description": "Single-halo physical-to-supercomoving conversion round-trip",
        "z_init": float(Z),
        "a_init": float(a),
        "n_grid": int(args.n_grid),
        "n_par": int(args.n_par),
        "pressure_scale": float(args.pressure_scale),
        "l_box_phys_kpc": float(l_box),
        "l_box_comoving_kpc": float(l_box / a),
        "dx_phys_kpc": float(l_box / args.n_grid),
        "dx_comoving_kpc": float((l_box / a) / args.n_grid),
        "r200_phys_kpc": float(ref.r200),
        "r200_comoving_kpc": float(ref.r200 / a),
    }

    with open(os.path.join(OUT_DIR, "stage3_params.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(OUT_DIR, "scalars.txt"), "w") as f:
        f.write("Stage 3 Scalar Diagnostics\n")
        for k, v in scalars.items():
            f.write(f"{k}: {v}\n")

    print("[Stage 3] Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
