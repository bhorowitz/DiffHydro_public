"""Shared utilities for hydro solvers."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array


def _as_idx(x: Any) -> Any:
    """Return the first element if ``x`` is a sequence, otherwise ``x`` itself."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def positivity_fix_with_eq(eq, U: Array, p_floor: float = 1e-12, rho_floor: float = 1e-12) -> Array:
    """Ensure positive density and pressure using the provided equation manager."""
    rho_i = _as_idx(eq.mass_ids)
    E_i = _as_idx(eq.energy_ids)
    prim = eq.get_primitives_from_conservatives(U)
    gamma = getattr(eq, "gamma", 1.4)

    rho = U[rho_i]
    v2 = 0.0
    for vid in getattr(eq, "vel_ids", []):
        v = prim[vid]
        v2 = v2 + v * v

    E = U[E_i]
    KE = 0.5 * rho * v2
    Emin = KE + p_floor / (gamma - 1.0)

    U = U.at[rho_i].set(jnp.maximum(rho, rho_floor))
    U = U.at[E_i].set(jnp.maximum(E, Emin))
    return U


def positivity_fix_simple(U: Array, rho_floor: float = 1e-12, e_floor: float = 1e-12) -> Array:
    """Fallback positivity fix when no equation manager is available."""
    U = U.at[0].set(jnp.maximum(U[0], rho_floor))
    if U.shape[0] >= 4:
        U = U.at[3].set(jnp.maximum(U[3], e_floor))
    return U


def positivity_fix(hydro, U: Array) -> Array:
    """Dispatch to the appropriate positivity fix for the given ``hydro`` solver."""
    eq = None
    if getattr(hydro, "fluxes", None):
        eq = getattr(hydro.fluxes[0], "eq_manage", None)
    if eq is not None:
        return positivity_fix_with_eq(eq, U)
    return positivity_fix_simple(U)


__all__ = [
    "Array",
    "positivity_fix",
    "positivity_fix_simple",
    "positivity_fix_with_eq",
]
