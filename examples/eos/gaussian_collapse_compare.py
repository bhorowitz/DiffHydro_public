from __future__ import annotations

import argparse
import numpy as np
import jax.numpy as jnp

from common import build_hydro, plot_comparison_grid, run_timeslices

import diffhydro as dh


def make_gaussian_collapse_ic(eq, *, nx: int, ny: int, p0: float):
    x = (jnp.arange(nx) + 0.5) / nx - 0.5
    y = (jnp.arange(ny) + 0.5) / ny - 0.5
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    sigma = 0.095
    rho = 1.0 + 4.0 * jnp.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma))
    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)

    # Set p ∝ rho so T = p/(rho*R) is spatially uniform at t=0.
    # Here p0 is the background pressure at rho=1.
    if eq.isothermal:
        p = eq.get_isothermal_pressure(rho)
    else:
        p = p0 * rho

    W = jnp.stack([rho, vx, vy, vz, p], axis=0)[..., None]
    return np.asarray(eq.get_conservatives_from_primitives(W))


def _log_density(U):
    rho = np.asarray(U[0, :, :, 0])
    return np.log10(np.maximum(rho, 1e-10))


def main():
    parser = argparse.ArgumentParser(description="Gaussian self-gravitating collapse: adiabatic vs isothermal.")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--G", type=float, default=1.0e-3)
    parser.add_argument("--p0", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="examples/isothermal/gaussian_collapse_compare.png")
    args = parser.parse_args()

    # In solver units (dx=1), self-gravity evolution is slow; use longer windows.
    times = [0.0, 10.0, 20.0]
    mesh_shape = [args.nx, args.ny, 1]

    eq_ad = dh.equationmanager.EquationManager(isothermal=False)
    eq_ad.mesh_shape = mesh_shape
    eq_ad.box_size = (1.0, 1.0, 1.0)

    p_ref = args.p0
    rho_ref = 1.0
    # Match isothermal c_s to the same initial temperature as adiabatic IC:
    # T0 = p0/(rho0*R), c_s^2 = R*T0 = p0/rho0.
    cs_ref = float(np.sqrt(p_ref / rho_ref))

    eq_iso = dh.equationmanager.EquationManager(
        isothermal=True,
        isothermal_sound_speed=cs_ref,
    )
    eq_iso.mesh_shape = mesh_shape
    eq_iso.box_size = (1.0, 1.0, 1.0)

    grav_ad = dh.gravity.MGSelfGravityForce(
        eq_ad,
        G=args.G,
        dx=1.0,
        subtract_mean=True,
        l=5,
        v1=2,
        v2=2,
        mu=1,
        iter_cycle=2,
        mg_tol=1.0e-6,
    )
    grav_iso = dh.gravity.MGSelfGravityForce(
        eq_iso,
        G=args.G,
        dx=1.0,
        subtract_mean=True,
        l=5,
        v1=2,
        v2=2,
        mu=1,
        iter_cycle=2,
        mg_tol=1.0e-6,
    )

    U0_ad = make_gaussian_collapse_ic(eq_ad, nx=args.nx, ny=args.ny, p0=args.p0)
    U0_iso = make_gaussian_collapse_ic(eq_iso, nx=args.nx, ny=args.ny, p0=args.p0)

    sim_ad = build_hydro(
        eq_ad,
        max_dt=0.2,
        n_super_step=8192,
        limiter="SUPERBEE",
        forces=[grav_ad],
    )
    sim_iso = build_hydro(
        eq_iso,
        max_dt=0.2,
        n_super_step=8192,
        limiter="SUPERBEE",
        forces=[grav_iso, dh.EOSProjectionForce(eq_iso)],
    )

    snaps_ad = run_timeslices(sim_ad, U0_ad, times)
    snaps_iso = run_timeslices(sim_iso, U0_iso, times)

    plot_comparison_grid(
        snaps_ad,
        snaps_iso,
        times,
        field_fn=_log_density,
        title="Gaussian Collapse log10(rho)",
        output_path=args.output,
        cmap="magma",
        include_temperature_rows=True,
        eq_adiabatic=eq_ad,
        eq_isothermal=eq_iso,
        temperature_cmap="inferno",
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
