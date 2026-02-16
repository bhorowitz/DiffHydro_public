import numpy as np
import jax.numpy as jnp

import diffhydro as dh


def _toy_state(shape=(8, 6, 1)):
    nx, ny, nz = shape
    x = jnp.linspace(0.0, 1.0, nx)[:, None, None]
    y = jnp.linspace(0.0, 1.0, ny)[None, :, None]
    z = jnp.linspace(0.0, 1.0, nz)[None, None, :]

    rho = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
    mx = 0.05 * rho * jnp.cos(2.0 * jnp.pi * y)
    my = 0.03 * rho * jnp.sin(2.0 * jnp.pi * x)
    mz = 0.01 * rho * (1.0 + z)

    # Deliberately inconsistent energy to test EOS projection.
    E = 10.0 + 0.1 * rho

    U = jnp.zeros((5, nx, ny, nz))
    U = U.at[0].set(rho)
    U = U.at[1].set(mx)
    U = U.at[2].set(my)
    U = U.at[3].set(mz)
    U = U.at[4].set(E)
    return U


def test_isothermal_primitives_use_cs2_rho():
    eq = dh.equationmanager.EquationManager(isothermal=True, isothermal_sound_speed=0.7)
    U = _toy_state()

    W = eq.get_primitives_from_conservatives(U)
    rho = np.asarray(W[eq.mass_ids])
    p = np.asarray(W[eq.energy_ids])

    p_expected = (eq.isothermal_sound_speed ** 2) * rho
    np.testing.assert_allclose(p, p_expected, rtol=1e-6, atol=1e-8)


def test_eos_projection_force_resets_energy_to_isothermal_state():
    eq = dh.equationmanager.EquationManager(isothermal=True, isothermal_sound_speed=0.5)
    projector = dh.EOSProjectionForce(eq)
    U = _toy_state()

    U_proj, _ = projector.force(0, U, {}, 0.1)

    rho = np.asarray(U_proj[eq.mass_ids])
    mx = np.asarray(U_proj[eq.vel_ids[0]])
    my = np.asarray(U_proj[eq.vel_ids[1]])
    mz = np.asarray(U_proj[eq.vel_ids[2]])
    E = np.asarray(U_proj[eq.energy_ids])

    kinetic = 0.5 * (mx * mx + my * my + mz * mz) / np.maximum(rho, eq.eps)
    p_iso = (eq.isothermal_sound_speed ** 2) * np.maximum(rho, eq.eps)
    thermal = p_iso / (eq.gamma - 1.0)
    E_expected = kinetic + thermal

    np.testing.assert_allclose(E, E_expected, rtol=1e-6, atol=1e-8)


def test_eos_projection_force_is_noop_for_non_isothermal():
    eq = dh.equationmanager.EquationManager(isothermal=False)
    projector = dh.EOSProjectionForce(eq)
    U = _toy_state()

    U_out, _ = projector.force(0, U, {}, 0.1)
    np.testing.assert_allclose(np.asarray(U_out), np.asarray(U), rtol=0.0, atol=0.0)
