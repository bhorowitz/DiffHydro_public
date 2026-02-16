from __future__ import annotations

import argparse
import numpy as np
import jax.numpy as jnp

from common import build_hydro, plot_comparison_grid, run_timeslices

import diffhydro as dh


def make_kh_initial_state(eq, *, nx: int, ny: int):
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Smooth shear interfaces around y=0.25 and y=0.75 reduce startup diffusion.
    y1 = 0.25
    y2 = 0.75
    width = 0.035
    shear = jnp.tanh((Y - y1) / width) - jnp.tanh((Y - y2) / width)

    vx = 0.5 * shear
    rho = jnp.ones_like(vx)

    # Localized vertical perturbation to seed KH billows.
    vy = 0.08 * jnp.sin(2.0 * jnp.pi * X) * (
        jnp.exp(-((Y - y1) / width) ** 2) + jnp.exp(-((Y - y2) / width) ** 2)
    )
    vz = jnp.zeros_like(vx)

    if eq.isothermal:
        p = eq.get_isothermal_pressure(rho)
    else:
        p = 1.0 * jnp.ones_like(rho)

    W = jnp.stack([rho, vx, vy, vz, p], axis=0)[..., None]
    return np.asarray(eq.get_conservatives_from_primitives(W))


def main():
    parser = argparse.ArgumentParser(description="Kelvin-Helmholtz: adiabatic vs isothermal.")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--output", type=str, default="examples/isothermal/kelvin_helmholtz_compare.png")
    args = parser.parse_args()

    # Note: solver uses dx=1 (cell units), so KH growth appears on O(10-100) times.
    times = [0.0, 80.0, 160.0]
    mesh_shape = [args.nx, args.ny, 1]

    eq_ad = dh.equationmanager.EquationManager(isothermal=False)
    eq_ad.mesh_shape = mesh_shape
    eq_ad.box_size = (1.0, 1.0, 1.0)

    p_ref = 1.0
    rho_ref = 1.0
    cs_ref = float(np.sqrt(eq_ad.gamma * p_ref / rho_ref))

    eq_iso = dh.equationmanager.EquationManager(
        isothermal=True,
        isothermal_sound_speed=cs_ref,
    )
    eq_iso.mesh_shape = mesh_shape
    eq_iso.box_size = (1.0, 1.0, 1.0)

    U0_ad = make_kh_initial_state(eq_ad, nx=args.nx, ny=args.ny)
    U0_iso = make_kh_initial_state(eq_iso, nx=args.nx, ny=args.ny)

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
        field_fn=lambda U: (
            (np.roll((U[2, :, :, 0] / np.maximum(U[0, :, :, 0], 1e-12)), -1, axis=0)
             - np.roll((U[2, :, :, 0] / np.maximum(U[0, :, :, 0], 1e-12)), 1, axis=0)) / 2.0
            - (np.roll((U[1, :, :, 0] / np.maximum(U[0, :, :, 0], 1e-12)), -1, axis=1)
               - np.roll((U[1, :, :, 0] / np.maximum(U[0, :, :, 0], 1e-12)), 1, axis=1)) / 2.0
        ),
        title="Kelvin-Helmholtz Vorticity (dv_y/dx - dv_x/dy)",
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
