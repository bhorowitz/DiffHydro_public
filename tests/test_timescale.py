import numpy as np
import jax
import jax.numpy as jnp

import diffhydro as dh
from diffhydro.physics.gravity import MGSelfGravityForce, FFTSelfGravityForce 
from diffhydro.prob_gen.blobs import make_gaussian_blob_rho 

def test_selfgravity_timestep_decreases_with_peak_density():
    shape = (16, 16, 16)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)

    # Two blobs with different peak density
    U_low = make_gaussian_blob_rho(eq, shape, peak_rho=10.0, background_rho=0.01, sigma=3.0)
    U_high = make_gaussian_blob_rho(eq, shape, peak_rho=40.0, background_rho=0.01, sigma=3.0)

    grav = MGSelfGravityForce(eq)

    dt_low = float(np.asarray(grav.timestep(U_low)))
    dt_high = float(np.asarray(grav.timestep(U_high)))

    # Higher peak density → smaller timestep
    assert dt_high < dt_low, f"dt_high={dt_high} should be < dt_low={dt_low}"
    # And "noticeably" smaller (guard against regressions in the formula)
    assert dt_high < 0.8 * dt_low
    
    grav = FFTSelfGravityForce(eq)


    dt_low_fft = float(np.asarray(grav.timestep(U_low)))
    dt_high_fft = float(np.asarray(grav.timestep(U_high)))

    # Higher peak density → smaller timestep
    assert dt_high_fft < dt_low_fft, f"dt_high={dt_high_fft} should be < dt_low={dt_low_fft}"
    # And "noticeably" smaller (guard against regressions in the formula)
    assert dt_high_fft < 0.8 * dt_low_fft