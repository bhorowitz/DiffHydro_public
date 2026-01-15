import jax.numpy as jnp
import numpy as onp
from diffhydro.hydro_core import hydro
from diffhydro.physics.gravity import FFTSelfGravityForce, gravity_accel_rfft
from diffhydro import comoving


class DummyEq:
    def __init__(self, mesh_shape):
        self.mesh_shape = tuple(mesh_shape)
        self.mass_ids = 0
        self.vel_ids = (1, 2, 3)
        self.energy_ids = 4


def make_hydro_with_force(mesh_shape=(4, 4, 4)):
    eq = DummyEq(mesh_shape)
    force = FFTSelfGravityForce(eq, G=1.0, subtract_mean=True)
    # minimal hydro: no fluxes, one force
    h = hydro(fluxes=[], forces=[force], use_mol=False)
    # Avoid running the sweep_stack (we only want forcing behavior)
    h.splitting_schemes = []
    return h, force


def test_hydro_advances_a_inside_params():
    h, force = make_hydro_with_force((4, 4, 4))
    nx, ny, nz = force.eq.mesh_shape
    nvars = 5
    U = jnp.zeros((nvars, nx, ny, nz), dtype=jnp.float32)
    params = {'a': 1.0, 'comoving': False}
    dt = 0.1

    (U2, params2), dt_out = h.hydrostep_adapt(0, (U, params), 0.0)
    # hydrostep_adapt uses internal dt calculation; to test _hydrostep directly,
    # call _hydrostep with chosen dt
    U3, params3 = h._hydrostep(0, (U, params), dt)
    assert 'a' in params3
    expected_a = comoving.step_a(1.0, dt, None)
    assert onp.allclose(onp.asarray(params3['a']), onp.asarray(expected_a))


def test_hydro_force_comoving_vs_noncomoving():
    h, force = make_hydro_with_force((4, 4, 4))
    nx, ny, nz = force.eq.mesh_shape
    nvars = 5

    # single-cell overdensity
    rho = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    rho = rho.at[0, 0, 0].set(3.0)
    U = jnp.zeros((nvars, nx, ny, nz), dtype=jnp.float32)
    U = U.at[0].set(rho)

    dt = 0.05
    a = 1.5

    # Non-comoving: params without comoving flag
    params_nc = {'a': a, 'comoving': False}
    U_nc, params_nc_out = h._hydrostep(0, (U, params_nc), dt)

    # Comoving: params with comoving=True
    params_c = {'a': a, 'comoving': True}
    U_c, params_c_out = h._hydrostep(0, (U, params_c), dt)

    # Compute ax from gravity_accel_rfft
    ax, ay, az = gravity_accel_rfft(rho, force.kx_r, force.ky_r, force.kz_r, force.k2_r, force.G, force.subtract_mean, a)

    # Expected momentum change: non-comoving Δm = ρ * ax * dt
    expected_nc = onp.asarray(rho) * onp.asarray(ax) * float(dt)
    # Comoving Δm = (ρ / a) * ax * dt
    expected_c = (onp.asarray(rho) / float(a)) * onp.asarray(ax) * float(dt)

    assert onp.allclose(onp.asarray(U_nc[1]), expected_nc, atol=1e-6)
    assert onp.allclose(onp.asarray(U_c[1]), expected_c, atol=1e-6)
