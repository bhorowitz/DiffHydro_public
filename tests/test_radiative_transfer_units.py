import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _import_stellar_radiation_force():
    candidates = [
        "diffhydro.physics.radiative_transfer",
        "diffhydro.radiativetransfer",
        "diffhydro.radiative_transfer",
        "radiativetransfer",
        "radiative_transfer",
    ]

    last_err = None
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "StellarRadiationForce"):
                return mod.StellarRadiationForce
        except Exception as err:
            last_err = err

    raise ImportError(
        "Impossible d'importer StellarRadiationForce. "
        "Adapte la liste des modules candidates dans _import_stellar_radiation_force(). "
        f"Dernière erreur: {last_err}"
    )


class DummyEq:
    def __init__(self, light_speed=3.0e10, eps=1e-30, mesh_shape=(16, 16, 16)):
        self.light_speed = light_speed
        self.eps = eps
        self.mesh_shape = mesh_shape


def _fmag(sol):
    return jnp.sqrt(sol[1] ** 2 + sol[2] ** 2 + sol[3] ** 2)


def _reduced_flux(sol, c, eps=1e-30):
    return _fmag(sol) / jnp.maximum(c * sol[0], eps)


@pytest.fixture
def eq():
    return DummyEq(light_speed=3.0e10, eps=1e-30, mesh_shape=(16, 16, 16))


@pytest.fixture
def rt(eq):
    StellarRadiationForce = _import_stellar_radiation_force()
    return StellarRadiationForce(
        escape_fraction=0.1,
        dx=1.0,
        injection_mode="stromgren",
        stromgren_rate=100.0,
        gaussian_star=True,
        injection_geometry="3D",
        injection_momentum=False,
        eq=eq,
        debug=False,
        momentum_only=False,
        beam_axis=0,
        beam_sign=1,
        beam_length_cells=4,
        beam_sigma=1.5,
        beam_reduced_flux=0.95,
    )


@pytest.fixture
def offsets_sigma():
    sigma = 1.5
    offsets = jnp.arange(-3, 4, dtype=jnp.int32)
    return offsets, sigma


def test_rt_state_layout_and_basic_invariants(eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    sol = sol.at[0, 3, 5, 7].set(2.0)
    sol = sol.at[1, 3, 5, 7].set(0.4 * eq.light_speed * 2.0)

    assert sol.shape == (4, nx, ny, nz)
    assert jnp.all(jnp.isfinite(sol))
    assert jnp.all(sol[0] >= 0.0)

    Fmag = _fmag(sol)
    assert jnp.all(Fmag <= eq.light_speed * sol[0] + 1e-30)

    f = _reduced_flux(sol, eq.light_speed, eq.eps)
    assert jnp.all(f >= 0.0)
    assert jnp.all(f <= 1.0 + 1e-12)


def test_clip_indices_2d(rt):
    x0, y0, z0 = 4, 5, 6
    di2, dj2 = jnp.meshgrid(jnp.array([-1, 0, 1]), jnp.array([-1, 0, 1]), indexing="ij")
    xi, yi, zi, valid = rt._clip_indices_2d(x0, y0, z0, di2, dj2)

    assert xi.shape == di2.shape
    assert yi.shape == dj2.shape
    assert zi.shape == di2.shape
    assert valid.shape == di2.shape
    assert jnp.all(zi == z0)
    assert jnp.all(jnp.isfinite(xi))
    assert jnp.all(jnp.isfinite(yi))
    assert jnp.all(jnp.isfinite(zi))


def test_clip_indices_3d(rt):
    x0, y0, z0 = 4, 5, 6
    di3, dj3, dk3 = jnp.meshgrid(jnp.array([-1, 0, 1]), jnp.array([-1, 0, 1]), jnp.array([-1, 0, 1]), indexing="ij")
    xi, yi, zi, valid = rt._clip_indices_3d(x0, y0, z0, di3, dj3, dk3)

    assert xi.shape == di3.shape
    assert yi.shape == dj3.shape
    assert zi.shape == dk3.shape
    assert valid.shape == di3.shape
    assert jnp.all(jnp.isfinite(xi))
    assert jnp.all(jnp.isfinite(yi))
    assert jnp.all(jnp.isfinite(zi))


def test_normalized_weights_2d_sum_to_one(rt):
    offsets = jnp.arange(-3, 4, dtype=jnp.int32)
    di2, dj2 = jnp.meshgrid(offsets, offsets, indexing="ij")
    valid = jnp.ones_like(di2, dtype=bool)

    _, _, w = rt._normalized_weights_2d(offsets, 1.5, valid)

    assert jnp.all(jnp.isfinite(w))
    assert np.isclose(float(jnp.sum(w)), 1.0, rtol=1e-12, atol=1e-12)
    assert jnp.all(w >= 0.0)


def test_normalized_weights_3d_sum_to_one(rt):
    offsets = jnp.arange(-2, 3, dtype=jnp.int32)
    di3, dj3, dk3 = jnp.meshgrid(offsets, offsets, offsets, indexing="ij")
    valid = jnp.ones_like(di3, dtype=bool)

    _, _, _, w = rt._normalized_weights_3d(offsets, 1.5, valid)

    assert jnp.all(jnp.isfinite(w))
    assert np.isclose(float(jnp.sum(w)), 1.0, rtol=1e-12, atol=1e-12)
    assert jnp.all(w >= 0.0)


# def test_rt_inject_momentum_beam_x_conserves_expected_total_fx(rt, eq):
#     nx, ny, nz = eq.mesh_shape
#     sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

#     x0, y0, z0 = 4, 8, 8
#     source = 1.0e5
#     sigma = 1.5
#     beam_len = 4

#     weights, sol_after = rt._inject_momentum_beam_x(
#         sol, x0, y0, z0, source, sigma, beam_len
#     )

#     assert jnp.all(jnp.isfinite(weights))
#     assert jnp.all(jnp.isfinite(sol_after))
#     assert np.isclose(float(jnp.sum(weights)), 1.0, rtol=1e-12, atol=1e-12)

#     assert np.isclose(float(jnp.sum(sol_after[2])), 0.0, atol=1e-30)
#     assert np.isclose(float(jnp.sum(sol_after[3])), 0.0, atol=1e-30)

#     expected_fx_sum = source * (eq.light_speed ** 2)
#     actual_fx_sum = float(jnp.sum(sol_after[1]))

#     assert np.isclose(actual_fx_sum, expected_fx_sum, rtol=1e-12), (
#         f"Somme Fx incohérente: attendu={expected_fx_sum:.16e}, obtenu={actual_fx_sum:.16e}"
#     )

#     assert np.isclose(float(jnp.sum(sol_after[0])), 0.0, atol=1e-30)


def test_rt_inject_energy_2d_conserves_expected_total_energy(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source = 7.5e12
    offsets = jnp.arange(-3, 4, dtype=jnp.int32)
    sigma = 1.5

    sol_after = rt._inject_energy_2d(sol, x0, y0, z0, source, offsets, sigma)

    assert jnp.all(jnp.isfinite(sol_after))
    actual_energy_sum = float(jnp.sum(sol_after[0]))

    assert np.isclose(actual_energy_sum, source, rtol=1e-12)

    assert np.isclose(float(jnp.sum(sol_after[1])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[2])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[3])), 0.0, atol=1e-30)


def test_rt_inject_energy_3d_conserves_expected_total_energy(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source = 7.5e12
    offsets = jnp.arange(-2, 3, dtype=jnp.int32)
    sigma = 1.5

    sol_after = rt._inject_energy_3d(sol, x0, y0, z0, source, offsets, sigma)

    assert jnp.all(jnp.isfinite(sol_after))
    actual_energy_sum = float(jnp.sum(sol_after[0]))

    assert np.isclose(actual_energy_sum, source, rtol=1e-12)

    assert np.isclose(float(jnp.sum(sol_after[1])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[2])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[3])), 0.0, atol=1e-30)


def test_rt_inject_momentum_x_2d(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source = 1.0e5
    offsets = jnp.arange(-3, 4, dtype=jnp.int32)
    sigma = 1.5

    weights2, sol_after = rt._inject_momentum_x_2d(sol, x0, y0, z0, source, offsets, sigma)

    assert jnp.all(jnp.isfinite(weights2))
    assert jnp.all(jnp.isfinite(sol_after))
    assert np.isclose(float(jnp.sum(weights2)), 1.0, rtol=1e-12, atol=1e-12)

    assert np.isclose(float(jnp.sum(sol_after[2])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[3])), 0.0, atol=1e-30)

    expected_fx_sum = source * (eq.light_speed ** 2)
    actual_fx_sum = float(jnp.sum(sol_after[1]))
    assert np.isclose(actual_fx_sum, expected_fx_sum, rtol=1e-12)


def test_rt_inject_momentum_x_3d(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source = 1.0e5
    offsets = jnp.arange(-2, 3, dtype=jnp.int32)
    sigma = 1.5

    sol_after = rt._inject_momentum_x_3d(sol, x0, y0, z0, source, offsets, sigma)

    assert jnp.all(jnp.isfinite(sol_after))
    assert np.isclose(float(jnp.sum(sol_after[2])), 0.0, atol=1e-30)
    assert np.isclose(float(jnp.sum(sol_after[3])), 0.0, atol=1e-30)

    expected_fx_sum = source * (eq.light_speed ** 2)
    actual_fx_sum = float(jnp.sum(sol_after[1]))
    assert np.isclose(actual_fx_sum, expected_fx_sum, rtol=1e-12)


def test_rt_clip_to_m1_cone_enforces_flux_bound(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    E0 = 2.5
    Fx0 = 10.0 * eq.light_speed * E0

    sol = sol.at[0, 5, 5, 5].set(E0)
    sol = sol.at[1, 5, 5, 5].set(Fx0)

    sol_clip = rt._clip_to_m1_cone(sol)

    assert jnp.all(jnp.isfinite(sol_clip))

    Fmag = _fmag(sol_clip)
    Fmax = rt.beam_reduced_flux * eq.light_speed * sol_clip[0]

    assert jnp.all(Fmag <= Fmax + 1e-12)

    clipped_value = float(Fmag[5, 5, 5])
    expected_value = float(rt.beam_reduced_flux * eq.light_speed * E0)

    assert np.isclose(clipped_value, expected_value, rtol=1e-12)


def test_rt_pipeline_energy_then_momentum_then_clip_is_finite_and_physical(rt, eq):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source_E = 1.0e5
    source_F = 1.0e5
    offsets = jnp.arange(-2, 3, dtype=jnp.int32)
    sigma = 1.5
    beam_len = 4

    sol = rt._inject_energy_3d(sol, x0, y0, z0, source_E, offsets, sigma)
    weights, sol = rt._inject_momentum_beam_x(sol, x0, y0, z0, source_F, sigma, beam_len)
    sol = rt._clip_to_m1_cone(sol)

    assert jnp.all(jnp.isfinite(sol))
    assert jnp.all(sol[0] >= 0.0)

    f = _reduced_flux(sol, eq.light_speed, eq.eps)
    assert jnp.all(f <= rt.beam_reduced_flux + 1e-12)

    active = np.array(sol[0] > eq.eps)
    assert active.any()

    max_f = float(jnp.max(f))
    assert max_f <= rt.beam_reduced_flux + 1e-12


def test_rt_diagnostic_print(rt, eq, capsys):
    nx, ny, nz = eq.mesh_shape
    sol = jnp.zeros((4, nx, ny, nz), dtype=jnp.float64)

    x0, y0, z0 = 4, 8, 8
    source_E = 1.0e5
    source_F = 1.0e5
    sigma = 1.5
    beam_len = 4

    offsets = jnp.arange(-2, 3, dtype=jnp.int32)

    sol = rt._inject_energy_3d(sol, x0, y0, z0, source_E, offsets, sigma)
    weights, sol = rt._inject_momentum_beam_x(sol, x0, y0, z0, source_F, sigma, beam_len)
    sol = rt._clip_to_m1_cone(sol)

    Fmag = _fmag(sol)
    f = _reduced_flux(sol, eq.light_speed, eq.eps)

    print("\n[RT DIAGNOSTIC]")
    print(f"sum(Egamma)           = {float(jnp.sum(sol[0])):.16e}")
    print(f"sum(Fx)               = {float(jnp.sum(sol[1])):.16e}")
    print(f"sum(Fy)               = {float(jnp.sum(sol[2])):.16e}")
    print(f"sum(Fz)               = {float(jnp.sum(sol[3])):.16e}")
    print(f"sum(weights)          = {float(jnp.sum(weights)):.16e}")
    print(f"max(Egamma)           = {float(jnp.max(sol[0])):.16e}")
    print(f"max(|F|)              = {float(jnp.max(Fmag)):.16e}")
    print(f"max(reduced_flux)     = {float(jnp.max(f)):.16e}")
    print(f"beam_reduced_flux     = {float(rt.beam_reduced_flux):.16e}")
    print(f"light_speed           = {float(eq.light_speed):.16e}")

    captured = capsys.readouterr()
    assert "[RT DIAGNOSTIC]" in captured.out