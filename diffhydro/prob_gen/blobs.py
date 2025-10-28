"""Blob-like initial conditions for conduction tests."""

from __future__ import annotations

import jax.numpy as jnp


def make_gaussian_blob(
    eq,
    shape,
    peak_temp: float = 10.0,
    background_temp: float = 1.0,
    sigma: float = 3.0,
):
    """Generate a 3-D Gaussian temperature perturbation in conservative form."""

    nx, ny, nz = shape
    sol = jnp.zeros((5, nx, ny, nz))
    rho0 = 1.0
    sol = sol.at[0].set(rho0)

    grid = [(jnp.arange(n) + 0.5) - 0.5 * n for n in (nx, ny, nz)]
    X, Y, Z = jnp.meshgrid(*grid, indexing="ij")
    r2 = X**2 + Y**2 + Z**2

    delta_T = peak_temp - background_temp
    temperature = background_temp + delta_T * jnp.exp(-r2 / (2.0 * sigma**2))

    pressure = rho0 * eq.R * temperature
    total_energy = eq.get_total_energy(
        pressure,
        rho0 * jnp.ones_like(pressure),
        jnp.zeros_like(pressure),
        jnp.zeros_like(pressure),
        jnp.zeros_like(pressure),
    )
    sol = sol.at[-1].set(total_energy)

    return sol, temperature


def make_gaussian_blob_rho(
    eq,
    shape,
    peak_rho: float = 10.0,
    background_rho: float = 0.01,
    sigma: float = 3.0,
):
    """Generate a 3-D Gaussian rho perturbation in conservative form."""

    nx, ny, nz = shape
    sol = jnp.zeros((5, nx, ny, nz))
    rho0 = 1.0
    sol = sol.at[-1].set(rho0)

    grid = [(jnp.arange(n) + 0.5) - 0.5 * n for n in (nx, ny, nz)]
    X, Y, Z = jnp.meshgrid(*grid, indexing="ij")
    r2 = X**2 + Y**2 + Z**2

    delta_rho = peak_rho - background_rho
    rho = background_rho + delta_rho * jnp.exp(-r2 / (2.0 * sigma**2))

    sol = sol.at[0].set(rho)

    return sol

__all__ = ["make_gaussian_blob"]

