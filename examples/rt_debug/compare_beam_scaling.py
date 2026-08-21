#!/usr/bin/env python
"""Compare RT beam momentum scalings after injection and after one hydro step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "2"))

import jax
import jax.numpy as jnp
import numpy as np

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat import (
    EquationManager as EquationManagerRT,
)
from diffhydro.physics.radiative_transfer import StellarRadiationForce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=int, default=16)
    parser.add_argument("--light-speed", type=float, default=2.0)
    parser.add_argument("--stromgren-rate", type=float, default=10.0)
    parser.add_argument("--beam-reduced-flux", type=float, default=0.95)
    parser.add_argument("--beam-length-cells", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs" / "compare_beam_scaling.json")
    return parser.parse_args()


def make_state_and_params(mesh_size: int):
    center = mesh_size // 2
    state = jnp.zeros((4, mesh_size, mesh_size, mesh_size), dtype=jnp.float32)
    params = {
        "star_masses": jnp.array([10.0], dtype=jnp.float32),
        "star_ages": jnp.array([0.1], dtype=jnp.float32),
        "star_metallicities": jnp.array([0.02], dtype=jnp.float32),
        "star_positions": jnp.array([[0, center, center]], dtype=jnp.int32),
    }
    return state, params


def max_reduced_flux(state, c: float) -> float:
    arr = np.asarray(jax.device_get(state))
    E = arr[0]
    Fmag = np.sqrt(arr[1] ** 2 + arr[2] ** 2 + arr[3] ** 2)
    ratio = np.where(E > 1e-30, Fmag / np.maximum(c * E, 1e-30), 0.0)
    return float(np.max(ratio))


def build_force_and_sim(args: argparse.Namespace, scaling: str):
    mesh_shape = (args.mesh_size, args.mesh_size, args.mesh_size)
    eq = EquationManagerRT(light_speed=args.light_speed, mesh_shape=mesh_shape)
    solver = dh.LaxFriedrichs_Radiative_transfer(
        equation_manager=eq,
        signal_speed=dh.signal_speed_Rusanov,
    )
    flux = dh.ConvectiveFlux_Radiative_transfer(eq, solver, dh.PLM(limiter="VANLEER"))
    force = StellarRadiationForce(
        escape_fraction=0.1,
        dx=1.0,
        injection_mode="stromgren",
        stromgren_rate=args.stromgren_rate,
        injection_momentum=True,
        injection_geometry="beam_x",
        eq=eq,
        debug=False,
        momentum_only=False,
        beam_axis=0,
        beam_sign=+1,
        beam_length_cells=args.beam_length_cells,
        beam_reduced_flux=args.beam_reduced_flux,
        beam_momentum_scaling=scaling,
    )
    sim = dh.hydro(n_super_step=1, fluxes=[flux], forces=[force])
    return force, sim


def run_case(args: argparse.Namespace, scaling: str) -> dict[str, float | str]:
    state0, params0 = make_state_and_params(args.mesh_size)
    force, sim = build_force_and_sim(args, scaling)
    dt = sim.timestep(state0)
    injected, _ = force.force(0, state0, params0, dt)
    (after_step, _), step_dt = sim.hydrostep_adapt(0, (state0, params0), 0.0)
    return {
        "scaling": scaling,
        "dt": float(jax.device_get(dt)),
        "step_dt": float(jax.device_get(step_dt)),
        "max_reduced_flux_after_injection": max_reduced_flux(injected, args.light_speed),
        "max_reduced_flux_after_one_step": max_reduced_flux(after_step, args.light_speed),
        "E_sum_after_injection": float(np.sum(np.asarray(jax.device_get(injected))[0])),
        "E_sum_after_one_step": float(np.sum(np.asarray(jax.device_get(after_step))[0])),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cases": [
            run_case(args, "physical"),
            run_case(args, "legacy_c2_source2"),
        ],
    }
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
