import jax.numpy as jnp
import numpy as onp
from diffhydro.physics.gravity import FFTSelfGravityForce, gravity_accel_rfft


class DummyEq:
    def __init__(self, mesh_shape):
        self.mesh_shape = tuple(mesh_shape)
        # variable layout: [rho, mx, my, mz, E]
        self.mass_ids = 0
        self.vel_ids = (1, 2, 3)
        self.energy_ids = 4


def test_fft_self_gravity_comoving_uniform_density():
    eq = DummyEq(mesh_shape=(4, 4, 4))
    G = 1.0
    force = FFTSelfGravityForce(eq, G=G, subtract_mean=True)

    nx, ny, nz = eq.mesh_shape
    nvars = 5
    U = jnp.zeros((nvars, nx, ny, nz), dtype=jnp.float32)
    # uniform comoving density -> zero source
    U = U.at[0].set(jnp.ones((nx, ny, nz), dtype=jnp.float32) * 2.0)
    dt = 0.1
    a = 2.0
    U_new, _ = force.force(0, U, {'a': a, 'comoving': True}, dt)

    # momentum should remain zero
    assert onp.allclose(onp.asarray(U_new[1]), onp.zeros_like(onp.asarray(U_new[1])))
    assert onp.allclose(onp.asarray(U_new[2]), onp.zeros_like(onp.asarray(U_new[2])))
    assert onp.allclose(onp.asarray(U_new[3]), onp.zeros_like(onp.asarray(U_new[3])))


def test_fft_self_gravity_comoving_single_overdensity_matches_ax():
    eq = DummyEq(mesh_shape=(4, 4, 4))
    G = 0.9
    force = FFTSelfGravityForce(eq, G=G, subtract_mean=True)

    nx, ny, nz = eq.mesh_shape
    nvars = 5
    U = jnp.zeros((nvars, nx, ny, nz), dtype=jnp.float32)
    rho = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    rho = rho.at[0, 0, 0].set(3.0)
    U = U.at[0].set(rho)

    dt = 0.05
    a = 1.5

    # compute ax from gravity_accel_rfft directly (should match internal call)
    ax, ay, az = gravity_accel_rfft(rho, force.kx_r, force.ky_r, force.kz_r, force.k2_r, force.G, force.subtract_mean, a)

    U_new, _ = force.force(0, U, {'a': a, 'comoving': True}, dt)

    # expected momentum change: Δm = (rho / a) * ax * dt
    expected_dm_x = (onp.asarray(rho) / float(a)) * onp.asarray(ax) * float(dt)
    assert onp.allclose(onp.asarray(U_new[1]), expected_dm_x, atol=1e-6)
