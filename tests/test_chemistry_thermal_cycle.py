"""End-to-end checks for hydrogen chemistry and thermal source terms.

These tests deliberately use the conservative combined state and exercise the
same forces that are assembled in a radiative-hydrodynamics run:

* photon absorption and H ionisation,
* photoheating from the excess photon energy,
* collisional/recombination/Compton cooling,
* explicit-Euler positivity limits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import diffhydro as dh
from diffhydro.equationmanager_radiative_transf_no_chat_copy import (
    EquationManager as EquationManagerRT,
)
from diffhydro.physics import hydrogen_chemistry as hchem
from diffhydro.physics.cooling import HeatCoolForce_basic
from diffhydro.physics.fraction_xHII import HydrogenPhotoChemistryForce
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits


N = 32
GAMMA = 5.0 / 3.0


@pytest.fixture
def system():
    cu = CodeUnits.from_config(
        {"length": "1.0 cm", "mass": "1 g", "velocity": "3e10 cm/s"},
        {"gamma": GAMMA, "mu": 0.61},
    )
    c_code = hchem.C_LIGHT_CGS / cu.V_cgs
    rt_eq = EquationManagerRT(
        light_speed=c_code,
        mesh_shape=(N, N, N),
        eps=1e-20,
        passive_weighted=False,
        passive_advected=False,
    )
    hydro_eq = dh.EquationManager(
        gamma=GAMMA,
        n_cons=6,
        passive_names=("x_HII",),
        mesh_shape=(N, N, N),
        eps=1e-20,
    )
    stellar = StellarRadiationForce(
        dx=1.0,
        injection_mode="stromgren",
        stromgren_rate=0.0,
        eq=rt_eq,
        hydro_eq=hydro_eq,
        cu=cu,
        chemistry=False,
        X_H=1.0,
    )
    return cu, c_code, stellar, rt_eq, hydro_eq


def make_state(cu, n_H, temperature_K, x_HII, photon_density):
    rho = n_H * hchem.MH_CGS / cu.rho_cgs
    pressure = n_H * (1.0 + x_HII) * hchem.KB_CGS * temperature_K / cu.P_cgs
    state = jnp.zeros((10, N, N, N), dtype=jnp.float64)
    state = state.at[0].set(photon_density * cu.L_cgs**3)
    state = state.at[4].set(rho)
    state = state.at[8].set(pressure / (GAMMA - 1.0))
    state = state.at[9].set(rho * x_HII)
    return state


def test_rate_table_and_cooling_components_are_finite():
    temperatures = jnp.array([100.0, 1.0e4, 1.0e6])
    coefficients = (
        hchem.beta_HI_cgs(temperatures),
        hchem.zeta_HI_cgs(temperatures),
        hchem.psi_HI_cgs(temperatures),
        hchem.alpha_A_HII_cgs(temperatures),
        hchem.alpha_B_HII_cgs(temperatures),
        hchem.eta_A_HII_cgs(temperatures),
        hchem.eta_B_HII_cgs(temperatures),
        hchem.theta_HII_cgs(temperatures),
    )
    for coefficient in coefficients:
        values = np.asarray(coefficient)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    cooling = hchem.cooling_rate_cgs(
        temperatures,
        n_HI=jnp.array([1.0, 0.5, 0.0]),
        n_HII=jnp.array([0.0, 0.5, 1.0]),
        n_e=jnp.array([0.0, 0.5, 1.0]),
    )
    assert np.all(np.isfinite(np.asarray(cooling)))
    assert np.all(np.asarray(cooling) >= 0.0)


def test_coupled_chemistry_heats_and_conserves_absorbed_photons(system):
    cu, _, stellar, _, _ = system
    chemistry = HydrogenPhotoChemistryForce(
        stellar,
        case="B",
        collisional=False,
        b_rec=0.0,
        include_heating=True,
        include_cooling=False,
        mean_photon_energy_eV=20.0,
    )
    state = make_state(cu, 1.0, 1.0e4, 0.0, 1.0e-4)
    view = chemistry.view
    photons_before = np.asarray(view.photon_density_cgs(state))
    energy_before = np.asarray(view.thermal_energy_code(state))
    ionised_before = np.asarray(view.xHII(state))

    state_after, _ = chemistry.force(0, state, {}, 1.0e8 / cu.T_cgs)
    photons_after = np.asarray(view.photon_density_cgs(state_after))
    energy_after = np.asarray(view.thermal_energy_code(state_after))
    ionised_after = np.asarray(view.xHII(state_after))

    absorbed = photons_before - photons_after
    np.testing.assert_allclose(
        absorbed,
        (ionised_after - ionised_before) * 1.0,
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        (energy_after - energy_before) * cu.P_cgs,
        absorbed * (20.0 - hchem.E_HI_EV) * hchem.EV_CGS,
        rtol=1e-8,
        atol=0.0,
    )
    assert np.all(photons_after >= 0.0)
    assert np.all((ionised_after >= 0.0) & (ionised_after <= 1.0))


def test_standalone_thermal_force_cools_without_making_energy_negative(system):
    cu, c_code, stellar, rt_eq, hydro_eq = system
    thermal = HeatCoolForce_basic(
        eq=rt_eq,
        hydro_eq=hydro_eq,
        cu=cu,
        light_speed=c_code,
        include_heating=False,
        include_cooling=True,
        X_H=1.0,
        max_frac=0.5,
    )
    state = make_state(cu, 1.0e3, 3.0e4, 1.0, 0.0)
    initial_temperature = float(thermal.get_temperature_K(state)[0, 0, 0])

    for step in range(12):
        state, _ = thermal.force(step, state, {}, 1.0e30)

    final_temperature = float(thermal.get_temperature_K(state)[0, 0, 0])
    energy = np.asarray(thermal.view.thermal_energy_code(state))
    assert final_temperature < initial_temperature
    assert np.all(np.isfinite(energy))
    assert np.all(energy > 0.0)
    assert float(thermal.net_rate_cgs(state)[0, 0, 0]) < 0.0


def test_thermal_source_signs_match_physical_regimes(system):
    cu, c_code, stellar, rt_eq, hydro_eq = system
    thermal = HeatCoolForce_basic(
        eq=rt_eq,
        hydro_eq=hydro_eq,
        cu=cu,
        light_speed=c_code,
        include_heating=True,
        include_cooling=True,
        mean_photon_energy_eV=20.0,
        X_H=1.0,
    )
    neutral = make_state(cu, 1.0, 1.0e4, 0.5, 1.0)
    ionised = make_state(cu, 1.0e3, 3.0e4, 1.0, 0.0)

    assert float(thermal.heating(neutral)[0, 0, 0]) > 0.0
    assert float(thermal.cooling(1.0e4, neutral)[0, 0, 0]) > 0.0
    assert float(thermal.net_rate_cgs(neutral)[0, 0, 0]) > 0.0
    assert float(thermal.net_rate_cgs(ionised)[0, 0, 0]) < 0.0
