# tests/test_conduction.py

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import diffhydro as dh


def make_gaussian_blob(eq, shape, peak_temp=10.0, background_temp=1.0, sigma=3.0):
    nx, ny, nz = shape
    # Conservative variables: [rho, rho*u, rho*v, rho*w, E]
    sol = jnp.zeros((5, nx, ny, nz))
    rho0 = 1.0
    sol = sol.at[0].set(rho0)

    grid = [
        (jnp.arange(n) + 0.5) - 0.5 * n
        for n in (nx, ny, nz)
    ]
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


def isotropy_score(field, nbins=36):
    nx, ny = field.shape
    x = (np.arange(nx) + 0.5) - 0.5 * nx
    y = (np.arange(ny) + 0.5) - 0.5 * ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    theta = (np.arctan2(Y, X) + 2 * np.pi) % (2 * np.pi)

    field_np = np.asarray(field)
    base = np.median(field_np)
    mask = field_np >= base + 0.1 * (field_np.max() - base)
    if not np.any(mask):
        return np.inf

    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    radii = []
    for i in range(nbins):
        sel = (theta >= edges[i]) & (theta < edges[i + 1]) & mask
        if np.any(sel):
            radii.append(np.percentile(r[sel], 95))
    if len(radii) < 3:
        return np.inf
    radii = np.asarray(radii)
    return float(np.std(radii) / (np.mean(radii) + 1e-12))


def blob_radius(field, background):
    nx, ny, nz = field.shape
    x = (np.arange(nx) + 0.5) - 0.5 * nx
    y = (np.arange(ny) + 0.5) - 0.5 * ny
    z = (np.arange(nz) + 0.5) - 0.5 * nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    threshold = background + 0.3 * (field.max() - background)
    mask = field >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.percentile(np.asarray(r)[mask], 95))


def test_conduction_blob_expansion():
    shape = (24, 24, 24)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.thermal_conductivity_model = "CUSTOM"
    eq.thermal_conductivity_fun = lambda T: jnp.ones_like(T) * 5.0

    init_sol, init_temperature = make_gaussian_blob(eq, shape)

    conductive_flux = dh.ConductiveFlux(eq, None, None)

    hydrosim = dh.hydro(
        n_super_step=120,
        fluxes=[conductive_flux],
        splitting_schemes=[[1, 2, 3]],
        maxjit=False,
    )

    state = hydrosim.evolve(init_sol, {})
    final_sol = state[0]

    init_primitives = eq.get_primitives_from_conservatives(init_sol)
    final_primitives = eq.get_primitives_from_conservatives(final_sol)

    init_temp = eq.get_temperature(
        init_primitives[eq.energy_ids], init_primitives[eq.mass_ids]
    )
    final_temp = eq.get_temperature(
        final_primitives[eq.energy_ids], final_primitives[eq.mass_ids]
    )

    initial_radius = blob_radius(np.asarray(init_temp), background=1.0)
    final_radius = blob_radius(np.asarray(final_temp), background=1.0)

    assert final_radius > 1.5 * initial_radius

    mid_plane = np.asarray(final_temp[:, :, shape[2] // 2])
    iso = isotropy_score(mid_plane)
    assert iso < 0.1, f"Solution not sufficiently isotropic: std/mean={iso:.3f}"
