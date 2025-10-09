"""Initial condition utilities for DiffHydro problem generation."""

from __future__ import annotations

import jax.numpy as jnp


def sedov(
    energy: float,
    ambient_density: float,
    conf,
):
    """Return conservative variables and characteristic time for a 3-D Sedov blast."""

    box_size = getattr(conf, "box_size", (1.0,) * len(conf.mesh_shape))
    cell_volume = (box_size[0] / conf.mesh_shape[0]) ** 3
    half_cells = int(conf.mesh_shape[0] / 2) - 1

    sol = jnp.zeros((5, conf.mesh_shape[0], conf.mesh_shape[1], conf.mesh_shape[2]))
    sol = sol.at[0].set(ambient_density)
    sol = sol.at[-1].set(1.0e-3)
    sol = sol.at[4, half_cells, half_cells, half_cells].set(energy / cell_volume)

    rmax = 3.0 * (box_size[0] / 2.0) / 4.0
    tf = jnp.sqrt((rmax / 1.15) ** 5 / energy)
    return sol, tf


def sedov_2d(
    energy: float,
    ambient_density: float,
    conf,
    embed_in_3d: bool = True,
):
    """Return conservative variables and characteristic time for a 2-D Sedov blast."""

    box_size = getattr(conf, "box_size", (1.0,) * len(conf.mesh_shape))
    cell_volume = (box_size[0] / conf.mesh_shape[0]) ** 3
    half_cells = int(conf.mesh_shape[0] / 2) - 1

    if embed_in_3d:
        nz = conf.mesh_shape[2] if len(conf.mesh_shape) > 2 else 1
        sol = jnp.zeros((5, conf.mesh_shape[0], conf.mesh_shape[1], nz))
        sol = sol.at[0].set(ambient_density)
        sol = sol.at[-1].set(1.0e-3)
        sol = sol.at[4, half_cells, half_cells, nz // 2].set(energy / cell_volume)
    else:
        sol = jnp.zeros((5, conf.mesh_shape[0], conf.mesh_shape[1]))
        sol = sol.at[0].set(ambient_density)
        sol = sol.at[-1].set(1.0e-3)
        sol = sol.at[4, half_cells, half_cells].set(energy / cell_volume)

    rmax = 3.0 * (box_size[0] / 2.0) / 4.0
    tf = jnp.sqrt((rmax / 1.15) ** 5 / energy)
    return sol, tf
