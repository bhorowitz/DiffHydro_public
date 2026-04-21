from __future__ import annotations

import jax.numpy as jnp


def gaussian_metallicity_blob(
    mesh_shape: tuple[int, int, int],
    *,
    amplitude: float,
    sigma_cells: float,
) -> jnp.ndarray:
    if float(amplitude) == 0.0:
        return jnp.zeros(mesh_shape, dtype=jnp.float32)

    nx, ny, nz = mesh_shape
    gx = jnp.arange(nx, dtype=jnp.float32)
    gy = jnp.arange(ny, dtype=jnp.float32)
    gz = jnp.arange(nz, dtype=jnp.float32)
    cx = 0.5 * jnp.asarray(nx - 1, dtype=jnp.float32)
    cy = 0.5 * jnp.asarray(ny - 1, dtype=jnp.float32)
    cz = 0.5 * jnp.asarray(nz - 1, dtype=jnp.float32)
    sigma2 = jnp.maximum(jnp.asarray(float(sigma_cells), dtype=jnp.float32) ** 2, 1.0e-6)

    dx = jnp.minimum(jnp.abs(gx - cx), nx - jnp.abs(gx - cx))
    dy = jnp.minimum(jnp.abs(gy - cy), ny - jnp.abs(gy - cy))
    dz = jnp.minimum(jnp.abs(gz - cz), nz - jnp.abs(gz - cz))
    r2 = dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2
    return jnp.asarray(float(amplitude), dtype=jnp.float32) * jnp.exp(-0.5 * r2 / sigma2)


def initialize_metal_density(
    rho_code: jnp.ndarray,
    *,
    metal_init: float,
    metal_floor: float = 0.0,
    blob_amplitude: float = 0.0,
    blob_sigma_cells: float = 4.0,
) -> jnp.ndarray:
    mesh_shape = tuple(int(s) for s in rho_code.shape)
    metal_fraction = jnp.ones_like(rho_code, dtype=jnp.float32) * jnp.asarray(float(metal_init), dtype=jnp.float32)
    if float(blob_amplitude) != 0.0:
        metal_fraction = metal_fraction + gaussian_metallicity_blob(
            mesh_shape,
            amplitude=float(blob_amplitude),
            sigma_cells=float(blob_sigma_cells),
        )
    metal_fraction = jnp.maximum(metal_fraction, jnp.asarray(float(metal_floor), dtype=jnp.float32))
    return jnp.maximum(rho_code, 0.0) * metal_fraction


def extract_metal_fields(U_code, eq) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    if not bool(getattr(eq, "has_metallicity", lambda: False)()):
        return None, None
    rho = jnp.maximum(U_code[eq.mass_ids], getattr(eq, "eps", 1.0e-20))
    rhoZ = jnp.maximum(U_code[eq.metal_density_ids], 0.0)
    Z = rhoZ / rho
    return rhoZ, Z
