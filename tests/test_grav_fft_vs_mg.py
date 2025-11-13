# tests/test_gravity_fft_vs_mg.py

import numpy as np
import jax
import jax.numpy as jnp

import diffhydro as dh
from diffhydro.physics.gravity import MGSelfGravityForce, FFTSelfGravityForce  # :contentReference[oaicite:12]{index=12}
from diffhydro.prob_gen.blobs import make_gaussian_blob_rho  # :contentReference[oaicite:13]{index=13}


def _accelerations_from_update(eq, U0, U1, dt):
    """Return vector acceleration field (3, nx, ny, nz) from Δmomentum."""
    rho = np.asarray(U0[eq.mass_ids])
    dU = np.asarray(U1 - U0)
    mom = dU[eq.vel_ids]  # momentum components in conservative vector
    return mom / (rho * dt)


def test_fft_and_mg_selfgravity_agree_on_small_blob():
    shape = (32, 32, 32)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)

    U = make_gaussian_blob_rho(eq, shape, peak_rho=10.0, background_rho=0.01, sigma=4.0)

    mg = MGSelfGravityForce(eq)
    fft = FFTSelfGravityForce(eq)

    dt = 1e-3

    U_mg = mg.force(0.0, U, {}, dt)
    U_fft = fft.force(0.0, U, {}, dt)

    a_mg = _accelerations_from_update(eq, U, U_mg, dt)
    a_fft = _accelerations_from_update(eq, U, U_fft, dt)

    rho = np.asarray(U[eq.mass_ids])
    mask = rho > 0.1 * rho.max()  # focus where the signal is non-trivial

    # L2 norm of difference / L2 norm of reference
    diff = (a_mg - a_fft)[mask]
    ref = a_fft[mask]

    num = np.sqrt((diff**2).sum()) + 1e-12
    den = np.sqrt((ref**2).sum()) + 1e-12
    rel_err = num / den

    assert rel_err < 0.1, f"MG vs FFT disagreement too large: rel_err={rel_err}"
