#!/usr/bin/env python
"""RT beam-injection reproducer extracted from examples/athena/blast-athena.ipynb.

The script intentionally skips the Athena/Sedov-Taylor blast setup. It starts
from a clean radiation state, injects from one star on the left x-boundary, and
saves diagnostics that make beam energy, momentum, and M1-cone violations easy
to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "2"))
os.environ.setdefault("DIFFHYDRO_DEBUG_CHECKS", "False")

import jax

jax.config.update("jax_disable_jit", False)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat import (
    EquationManager as EquationManagerRT,
)
from diffhydro.physics.radiative_transfer import StellarRadiationForce


FIELD_NAMES = ("E_gamma", "F_gamma_x", "F_gamma_y", "F_gamma_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and plot the RT beam-injection debug case."
    )
    parser.add_argument("--mesh-size", type=int, default=100)
    parser.add_argument("--t-target", type=float, default=18.6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-super-step", type=int, default=1000)
    parser.add_argument("--light-speed", type=float, default=2.0)
    parser.add_argument("--stromgren-rate", type=float, default=10.0)
    parser.add_argument("--beam-length-cells", type=int, default=1)
    parser.add_argument("--beam-reduced-flux", type=float, default=0.95)
    parser.add_argument(
        "--beam-momentum-scaling",
        default="physical",
        choices=("physical", "legacy_c2_source2"),
    )
    parser.add_argument("--limiter", default="VANLEER")
    parser.add_argument(
        "--rt-solver",
        default="rusanov",
        choices=("rusanov", "hll-local"),
    )
    parser.add_argument("--periodic-flux-divergence", action="store_true")
    parser.add_argument("--geometry", default="beam_x", choices=("beam_x", "2D", "3D"))
    parser.add_argument("--no-momentum", action="store_true")
    parser.add_argument("--debug-force", action="store_true")
    parser.add_argument("--debug-fixed-dt", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--star-x", type=int, default=0)
    parser.add_argument("--star-y", type=int, default=None)
    parser.add_argument("--star-z", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    return parser.parse_args()


def make_output_dirs(output_root: Path, run_name: str | None) -> dict[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / (run_name or f"beam_injection_{stamp}")
    dirs = {
        "run": run_dir,
        "slices": run_dir / "images" / "slices",
        "profiles": run_dir / "images" / "profiles",
        "metrics": run_dir / "images" / "metrics",
        "data": run_dir / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_sim(args: argparse.Namespace) -> dh.hydro:
    mesh_shape = (args.mesh_size, args.mesh_size, args.mesh_size)
    eq = EquationManagerRT(
        light_speed=args.light_speed,
        mesh_shape=mesh_shape,
        debug=args.debug_force,
    )
    if args.rt_solver == "hll-local":
        solver = dh.HLL_Radiative_transfer_Local(equation_manager=eq)
    else:
        solver = dh.LaxFriedrichs_Radiative_transfer(
            equation_manager=eq,
            signal_speed=dh.signal_speed_Rusanov,
        )
    flux = dh.ConvectiveFlux_Radiative_transfer(
        eq,
        solver,
        dh.PLM(limiter=args.limiter),
    )
    stellar_force = StellarRadiationForce(
        escape_fraction=0.1,
        dx=1.0,
        injection_mode="stromgren",
        stromgren_rate=args.stromgren_rate,
        injection_momentum=not args.no_momentum,
        injection_geometry=args.geometry,
        eq=eq,
        debug=args.debug_force,
        momentum_only=False,
        beam_axis=0,
        beam_sign=+1,
        beam_length_cells=args.beam_length_cells,
        beam_reduced_flux=args.beam_reduced_flux,
        beam_momentum_scaling=args.beam_momentum_scaling,
    )
    return dh.hydro(
        n_super_step=args.n_super_step,
        fluxes=[flux],
        forces=[stellar_force],
        debug_fixed_dt=args.debug_fixed_dt,
        periodic_flux_divergence=args.periodic_flux_divergence,
    )


def initial_state_and_params(args: argparse.Namespace) -> tuple[jax.Array, dict[str, jax.Array]]:
    center = args.mesh_size // 2
    star_y = center if args.star_y is None else args.star_y
    star_z = center if args.star_z is None else args.star_z
    state = jnp.zeros((4, args.mesh_size, args.mesh_size, args.mesh_size), dtype=jnp.float32)
    params = {
        "star_masses": jnp.array([10.0], dtype=jnp.float32),
        "star_ages": jnp.array([0.1], dtype=jnp.float32),
        "star_metallicities": jnp.array([0.02], dtype=jnp.float32),
        "star_positions": jnp.array([[args.star_x, star_y, star_z]], dtype=jnp.int32),
    }
    return state, params


def host_array(x: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def m1_ratio(state: np.ndarray, light_speed: float, eps: float = 1e-30) -> np.ndarray:
    energy = state[0]
    flux_mag = np.sqrt(state[1] ** 2 + state[2] ** 2 + state[3] ** 2)
    return np.where(energy > eps, flux_mag / np.maximum(light_speed * energy, eps), 0.0)


def finite_log10(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(np.abs(values), floor))


def save_slice(
    arr: np.ndarray,
    path: Path,
    title: str,
    xlabel: str = "y",
    ylabel: str = "x",
    cmap: str = "magma",
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_axis_profiles(state: np.ndarray, ratio: np.ndarray, args: argparse.Namespace, out_dir: Path) -> None:
    center = args.mesh_size // 2
    x = np.arange(args.mesh_size)
    profiles = {
        "E_gamma": state[0, :, center, center],
        "F_gamma_x": state[1, :, center, center],
        "F_gamma_y": state[2, :, center, center],
        "F_gamma_z": state[3, :, center, center],
        "m1_ratio": ratio[:, center, center],
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("E_gamma", "F_gamma_x", "F_gamma_y", "F_gamma_z"):
        ax.plot(x, profiles[name], label=name)
    ax.set_title("Beam-axis fields at y=z=center")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "beam_axis_fields.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, finite_log10(profiles["E_gamma"]), label="log10 |E_gamma|")
    ax.plot(x, finite_log10(profiles["F_gamma_x"]), label="log10 |F_gamma_x|")
    ax.set_title("Beam-axis log profiles")
    ax.set_xlabel("x")
    ax.set_ylabel("log10(abs(value))")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "beam_axis_log_energy_fx.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, profiles["m1_ratio"], label="|F| / (c E)")
    ax.axhline(args.beam_reduced_flux, color="tab:red", linestyle="--", label="beam_reduced_flux")
    ax.axhline(1.0, color="black", linestyle=":", label="M1 limit")
    ax.set_title("M1 cone diagnostic on beam axis")
    ax.set_xlabel("x")
    ax.set_ylabel("|F| / (c E)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "beam_axis_m1_ratio.png", dpi=180)
    plt.close(fig)

    np.savez(out_dir.parent.parent / "data" / "beam_axis_profiles.npz", x=x, **profiles)


def save_diagnostics(
    state: np.ndarray,
    dt_hist: np.ndarray,
    n_steps: int,
    t_final: float,
    args: argparse.Namespace,
    dirs: dict[str, Path],
) -> dict[str, float | int | bool | str | list[str]]:
    center = args.mesh_size // 2
    ratio = m1_ratio(state, args.light_speed)
    active_dt = dt_hist[:n_steps]

    metrics = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "mesh_size": args.mesh_size,
        "geometry": args.geometry,
        "rt_solver": args.rt_solver,
        "periodic_flux_divergence": args.periodic_flux_divergence,
        "light_speed": args.light_speed,
        "stromgren_rate": args.stromgren_rate,
        "beam_length_cells": args.beam_length_cells,
        "beam_reduced_flux": args.beam_reduced_flux,
        "beam_momentum_scaling": args.beam_momentum_scaling,
        "injection_momentum": not args.no_momentum,
        "star_position": [
            args.star_x,
            args.star_y if args.star_y is not None else args.mesh_size // 2,
            args.star_z if args.star_z is not None else args.mesh_size // 2,
        ],
        "t_final": t_final,
        "n_steps": n_steps,
        "dt_min": float(np.min(active_dt)) if active_dt.size else 0.0,
        "dt_max": float(np.max(active_dt)) if active_dt.size else 0.0,
        "E_sum": float(np.sum(state[0])),
        "E_max": float(np.max(state[0])),
        "Fx_sum": float(np.sum(state[1])),
        "Fx_max": float(np.max(state[1])),
        "Fy_abs_max": float(np.max(np.abs(state[2]))),
        "Fz_abs_max": float(np.max(np.abs(state[3]))),
        "m1_ratio_max": float(np.max(ratio)),
        "m1_ratio_gt_reduced_flux": bool(np.any(ratio > args.beam_reduced_flux)),
        "m1_ratio_gt_one": bool(np.any(ratio > 1.0)),
    }

    with (dirs["run"] / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    np.savez_compressed(
        dirs["data"] / "final_state.npz",
        state=state,
        dt_hist=active_dt,
        m1_ratio=ratio,
    )

    for idx, name in enumerate(FIELD_NAMES):
        plane = state[idx, :, :, center]
        save_slice(
            plane,
            dirs["slices"] / f"{idx}_{name}_z_center.png",
            f"{name} at z={center}",
        )
        save_slice(
            finite_log10(plane),
            dirs["slices"] / f"{idx}_{name}_z_center_log10_abs.png",
            f"log10(abs({name})) at z={center}",
        )

    window = max(5, min(args.mesh_size - 1, args.beam_length_cells + 8))
    save_slice(
        state[0, :, :, center],
        dirs["slices"] / "E_gamma_beam_zoom.png",
        f"E_gamma beam zoom at z={center}",
        xlim=(max(center - 8, 0), min(center + 8, args.mesh_size - 1)),
        ylim=(0, min(window, args.mesh_size - 1)),
    )
    save_slice(
        ratio[:, :, center],
        dirs["metrics"] / "m1_ratio_z_center.png",
        f"|F|/(c E) at z={center}",
        cmap="viridis",
    )

    save_axis_profiles(state, ratio, args, dirs["profiles"])
    return metrics


def main() -> None:
    args = parse_args()
    dirs = make_output_dirs(args.output_root, args.run_name)

    print("RT beam-injection debug run")
    print(f"repo: {REPO_ROOT}")
    print(f"output: {dirs['run']}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"backend={jax.default_backend()}")
    print(f"devices={jax.devices()}")

    sim = build_sim(args)
    state0, params = initial_state_and_params(args)
    final_state, _, t_final, dt_hist, n_steps = sim.evolve_till_time(
        state0,
        params,
        args.t_target,
        max_steps=args.max_steps,
    )

    state_np = host_array(final_state)
    dt_hist_np = host_array(dt_hist)
    n_steps_int = int(host_array(n_steps))
    t_final_float = float(host_array(t_final))
    metrics = save_diagnostics(
        state_np,
        dt_hist_np,
        n_steps_int,
        t_final_float,
        args,
        dirs,
    )

    print(json.dumps(metrics, indent=2))
    print(f"saved diagnostics under {dirs['run']}")


if __name__ == "__main__":
    main()
