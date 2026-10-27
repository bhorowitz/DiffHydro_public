from __future__ import annotations

import argparse
import numpy as np
import jax.numpy as jnp

from common import build_hydro, plot_comparison_grid, run_timeslices

import diffhydro as dh


def make_linear_wave_ic(eq, *, nx: int, ny: int, amplitude: float, cs_target: float):
    x = (jnp.arange(nx) + 0.5) / nx
    X = x[:, None]
    # Two-mode perturbation makes phase propagation visually obvious in snapshots.
    phase1 = 2.0 * jnp.pi * X
    phase2 = 4.0 * jnp.pi * X

    rho0 = 1.0
    drho = amplitude * rho0 * (jnp.sin(phase1) + 0.5 * jnp.sin(phase2))
    rho = rho0 + drho

    if eq.isothermal:
        vx = amplitude * cs_target * (jnp.sin(phase1) + 0.5 * jnp.sin(phase2))
        p = eq.get_isothermal_pressure(rho)
    else:
        p0 = (cs_target * cs_target) * rho0 / eq.gamma
        dp = (cs_target * cs_target) * drho
        vx = dp / (rho0 * cs_target)
        p = p0 + dp

    vy = jnp.zeros_like(vx)
    vz = jnp.zeros_like(vx)

    rho = jnp.tile(rho[:, :, None], (1, ny, 1))
    vx = jnp.tile(vx[:, :, None], (1, ny, 1))
    vy = jnp.tile(vy[:, :, None], (1, ny, 1))
    vz = jnp.tile(vz[:, :, None], (1, ny, 1))
    p = jnp.tile(p[:, :, None], (1, ny, 1))

    W = jnp.stack([rho, vx, vy, vz, p], axis=0)
    return np.asarray(eq.get_conservatives_from_primitives(W))


def _density_perturbation(U):
    rho = np.asarray(U[0, :, :, 0])
    return rho - rho.mean()


def main():
    parser = argparse.ArgumentParser(description="Linear sound waves: adiabatic vs isothermal.")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--amplitude", type=float, default=1e-2)
    parser.add_argument("--output", type=str, default="examples/isothermal/linear_sound_wave_compare.png")
    args = parser.parse_args()

    # With dx=1 in solver units, a wavelength of nx cells has period ~nx/c_s.
    times = [0.0, 32.0, 64.0]
    mesh_shape = [args.nx, args.ny, 1]
    cs_target = 1.0

    eq_ad = dh.equationmanager.EquationManager(isothermal=False)
    eq_ad.mesh_shape = mesh_shape
    eq_ad.box_size = (1.0, 1.0, 1.0)

    eq_iso = dh.equationmanager.EquationManager(
        isothermal=True,
        isothermal_sound_speed=cs_target,
    )
    eq_iso.mesh_shape = mesh_shape
    eq_iso.box_size = (1.0, 1.0, 1.0)

    U0_ad = make_linear_wave_ic(eq_ad, nx=args.nx, ny=args.ny, amplitude=args.amplitude, cs_target=cs_target)
    U0_iso = make_linear_wave_ic(eq_iso, nx=args.nx, ny=args.ny, amplitude=args.amplitude, cs_target=cs_target)

    sim_ad = build_hydro(
        eq_ad,
        max_dt=0.2,
        n_super_step=8192,
        limiter="SUPERBEE",
    )
    sim_iso = build_hydro(
        eq_iso,
        max_dt=0.2,
        n_super_step=8192,
        limiter="SUPERBEE",
        forces=[dh.EOSProjectionForce(eq_iso)],
    )

    snaps_ad = run_timeslices(sim_ad, U0_ad, times)
    snaps_iso = run_timeslices(sim_iso, U0_iso, times)

    plot_comparison_grid(
        snaps_ad,
        snaps_iso,
        times,
        field_fn=_density_perturbation,
        title="Linear Sound Wave: rho - <rho>",
        output_path=args.output,
        cmap="RdBu_r",
        include_temperature_rows=True,
        eq_adiabatic=eq_ad,
        eq_isothermal=eq_iso,
        temperature_cmap="inferno",
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
