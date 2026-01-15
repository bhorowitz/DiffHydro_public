"""Small helpers for comoving-coordinate support.

These helpers are intentionally lightweight and pure-function so they can be
used inside JAX transformations or unit tests without touching the larger
hydro/evolution code yet. They wrap jax_cosmo for H(a) when available.
"""
from __future__ import annotations
import jax.numpy as jnp
import jax
try:
    import jax_cosmo as jc
except Exception:  # pragma: no cover - if jax_cosmo missing, fall back to simple matter-dom
    jc = None


def hubble(cosmo, a):
    """Return H(a) using jax_cosmo if available, otherwise matter-dominated H0*a^{-3/2}.

    This function is JAX-friendly (uses jnp) so it can be used inside jitted
    code paths. `cosmo` may be a jax_cosmo cosmology object or a scalar H0.
    """
    a = jnp.asarray(a)
    # If jax_cosmo is available and a cosmology object was passed, use it.
    if jc is not None and cosmo is not None:
        return jc.background.hubble(cosmo, a)

    # Support a simple dict-based cosmology: {'H0': float, 'Omega_m': float}
    # for a flat LambdaCDM: H(a) = H0 * sqrt(Omega_m * a^-3 + (1 - Omega_m)).
    if isinstance(cosmo, dict):
        H0 = jnp.asarray(cosmo.get('H0', 1.0))
        Om = jnp.asarray(cosmo.get('Omega_m', 1.0))
        return H0 * jnp.sqrt(Om * a ** (-3.0) + (1.0 - Om))

    # If cosmo is a scalar, treat as H0 and assume matter dominated
    H0 = jnp.asarray(cosmo) if cosmo is not None else jnp.asarray(1.0)
    return H0 * a ** (-1.5)


def step_a(a, dt, cosmo=None):
    """Advance scale factor by dt (physical time) with da/dt = a H(a).

    Uses an explicit Euler step. Returns the new scale factor as a JAX array.
    """
    a = jnp.asarray(a)
    dt = jnp.asarray(dt)
    H = hubble(cosmo, a)
    return a + dt * a * H


def poisson_rhs_comoving(rho_c: jnp.ndarray, a, G: float = 1.0, subtract_mean: bool = True):
    """Return Poisson RHS in comoving coordinates.

    Equation: ∇_x^2 φ = 4π G / a * (rho_c - ⟨rho_c⟩)
    Returns F = (4πG / a) * (rho_c - mean).
    """
    rho_c = jnp.asarray(rho_c)
    mean = jnp.mean(rho_c)
    src = rho_c - jnp.where(subtract_mean, mean, 0.0)
    return (4.0 * jnp.pi * jnp.asarray(G) / jnp.asarray(a)) * src


def hubble_momentum_kick(momentum: jnp.ndarray, H, dt):
    """Apply Hubble drag to comoving momentum: d(rho u)/dt = -H rho u.

    Uses first-order update: m -> m * (1 - H dt).
    """
    return momentum * (1.0 - jnp.asarray(H) * jnp.asarray(dt))


def hubble_energy_damp(E_kin: jnp.ndarray, H, dt):
    """Damp kinetic energy by Hubble expansion: dE_kin/dt = -2 H E_kin.

    Uses first-order update E -> E * (1 - 2 H dt).
    """
    return E_kin * (1.0 - 2.0 * jnp.asarray(H) * jnp.asarray(dt))
