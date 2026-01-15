import numpy as onp
import jax.numpy as jnp
from diffhydro import comoving


def test_step_a_numeric_H0():
    a0 = 1.0
    H0 = 2.0
    dt = 0.1
    a1 = comoving.step_a(a0, dt, cosmo=H0)
    # explicit Euler: a1 = a0 + dt * a0 * H0 = 1 + 0.1 * 1 * 2 = 1.2
    assert abs(a1 - 1.2) < 1e-12


def test_poisson_rhs_comoving_constant():
    # uniform comoving density -> zero RHS (mean subtraction)
    rho = jnp.ones((4, 4), dtype=jnp.float32) * 2.0
    rhs = comoving.poisson_rhs_comoving(rho, a=2.0, G=1.0, subtract_mean=True)
    assert onp.allclose(onp.asarray(rhs), onp.zeros_like(onp.asarray(rhs)))


def test_poisson_rhs_comoving_contrast():
    # single-cell overdensity
    rho = jnp.ones((2, 2), dtype=jnp.float32)
    rho = rho.at[0, 0].set(3.0)
    a = 0.5
    G = 0.7
    rhs = comoving.poisson_rhs_comoving(rho, a=a, G=G, subtract_mean=True)
    mean = float(jnp.mean(rho))
    expected = (4.0 * jnp.pi * G / a) * (rho - mean)
    assert onp.allclose(onp.asarray(rhs), onp.asarray(expected))


def test_hubble_momentum_and_energy_kick():
    mom = jnp.array([[1.0, -2.0], [0.5, 0.0]], dtype=jnp.float32)
    H = 0.3
    dt = 0.2
    mom2 = comoving.hubble_momentum_kick(mom, H, dt)
    expected_mom2 = mom * (1.0 - H * dt)
    assert onp.allclose(onp.asarray(mom2), onp.asarray(expected_mom2))

    E = jnp.array([1.0, 2.0, 0.5], dtype=jnp.float32)
    E2 = comoving.hubble_energy_damp(E, H, dt)
    expected_E2 = E * (1.0 - 2.0 * H * dt)
    assert onp.allclose(onp.asarray(E2), onp.asarray(expected_E2))
