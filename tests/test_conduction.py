# tests/test_conduction.py

import numpy as np
import pytest

import jax
jnp = jax.numpy

import diffhydro as dh
from diffhydro.prob_gen import make_gaussian_blob
from diffhydro.utils.diagnostics import blob_radius, isotropy_score


def test_conduction_blob_expansion():
    shape = (24, 24, 24)
    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = list(shape)
    eq.thermal_conductivity_model = "SUTHERLAND"
    eq._set_transport_properties(None)
 #   eq.thermal_conductivity_fun = lambda T: jnp.ones_like(T) * 5.0

    init_sol, init_temperature = make_gaussian_blob(eq, shape)

    conductive_flux = dh.ConductiveFlux(eq, None, None)

    hydrosim = dh.hydro(
        n_super_step=120,
        fluxes=[conductive_flux],
    #    splitting_schemes=[[1, 2, 3]],
        maxjit=False,
    )

    state = hydrosim.evolve(init_sol, {})
    final_sol = state[0]

    init_primitives = eq.get_primitives_from_conservatives(init_sol)
    final_primitives = eq.get_primitives_from_conservatives(final_sol)

    init_temp = eq.get_temperature(
        init_primitives[eq.energy_ids], init_primitives[eq.mass_ids]
    )
    final_temp = eq.get_temperature(
        final_primitives[eq.energy_ids], final_primitives[eq.mass_ids]
    )

    initial_radius = blob_radius(np.asarray(init_temp), background=1.0)
    final_radius = blob_radius(np.asarray(final_temp), background=1.0)

    assert final_radius > 1.5 * initial_radius

    mid_plane = np.asarray(final_temp[:, :, shape[2] // 2])
    iso = isotropy_score(mid_plane)
    assert iso < 0.1, f"Solution not sufficiently isotropic: std/mean={iso:.3f}"
