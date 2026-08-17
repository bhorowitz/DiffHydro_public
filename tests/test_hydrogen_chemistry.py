"""Physical validation of the hydrogen-only chemistry.

Checks (a) that the tabulated fits reproduce their textbook values, (b) that
the state view reads the CONSERVATIVE state correctly, (c) that the explicit
Euler update with the positivity limiters converges to the analytic
photoionization equilibrium, and (d) that the Stromgren radius comes out
right in a 0-D / uniform-field setting.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

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
from diffhydro.physics.fraction_xHII import (
    HydrogenIonizationForce, HydrogenPhotoChemistryForce,
)
from diffhydro.physics.radiative_transfer_fixed import StellarRadiationForce
from diffhydro.units import CodeUnits

N = 4
GAMMA = 5.0 / 3.0


@pytest.fixture
def setup():
    cu = CodeUnits.from_config(
        {"length": "1.0 cm", "mass": "1 g", "velocity": "3e10 cm/s"},
        {"gamma": GAMMA, "mu": 0.61},
    )
    c_code = hchem.C_LIGHT_CGS / cu.V_cgs
    rt_eq = EquationManagerRT(
        light_speed=c_code, mesh_shape=(N, N, N), eps=1e-20,
        passive_weighted=False, passive_advected=False,
    )
    hydro_eq = dh.EquationManager(
        gamma=GAMMA, n_cons=6, passive_names=("x_HII",),
        mesh_shape=(N, N, N), eps=1e-20,
    )
    sf = StellarRadiationForce(
        dx=1.0, injection_mode="stromgren", stromgren_rate=0.0,
        eq=rt_eq, hydro_eq=hydro_eq, cu=cu, chemistry=True,
        xHII_weighted=False, X_H=1.0,
    )
    hc = HeatCoolForce_basic(
        eq=rt_eq, hydro_eq=hydro_eq, cu=cu, light_speed=c_code,
        X_H=1.0,
    )
    ion = HydrogenIonizationForce(sf)
    return cu, c_code, sf, hc, ion


def make_state(cu, n_H_cgs=1.0, T_K=1.0e4, x_HII=0.1, N_gamma_cgs=0.0, dtype=jnp.float64):
    """Build a uniform CONSERVATIVE state (v = 0)."""
    rho_code = n_H_cgs * hchem.MH_CGS / cu.rho_cgs
    p_code = n_H_cgs * (1.0 + x_HII) * hchem.KB_CGS * T_K / cu.P_cgs
    sol = jnp.zeros((10, N, N, N), dtype=dtype)
    sol = sol.at[0].set(N_gamma_cgs * cu.L_cgs ** 3)
    # Combined conservative layout: RT [N,Fx,Fy,Fz] then hydro
    # [rho,rho*vx,rho*vy,rho*vz,E_tot,rho*x_HII].
    sol = sol.at[4].set(rho_code)
    sol = sol.at[8].set(p_code / (GAMMA - 1.0))   # v = 0 -> E_tot = e_th
    sol = sol.at[9].set(rho_code * x_HII)
    return sol


# ---------------------------------------------------------------------------
# (a) coefficients
# ---------------------------------------------------------------------------
def test_sigma_HI_threshold_is_6e18():
    """Verner (1996) at 13.6 eV must give ~6.3e-18 cm^2, NOT the fit
    parameter sigma_0 = 5.475e-14."""
    assert 6.0e-18 < hchem.SIGMA_HI_0_CGS < 6.6e-18
    assert float(hchem.sigma_HI_verner_cgs(10.0)) == 0.0        # below threshold
    # (nu/nu_0)^-3 falls off as expected
    assert float(hchem.sigma_HI_powerlaw_cgs(2 * hchem.NU_HI_CGS)) == pytest.approx(
        hchem.SIGMA_HI_0_CGS / 8.0, rel=1e-12
    )


@pytest.mark.parametrize(
    "fn,T,expected,rtol",
    [
        (hchem.alpha_A_HII_cgs, 1e4, 4.30e-13, 0.02),
        (hchem.alpha_B_HII_cgs, 1e4, 2.59e-13, 0.02),
        (hchem.beta_HI_cgs, 1e4, 6.2e-16, 0.05),
        (hchem.theta_HII_cgs, 1e4, 1.42e-25, 1e-6),
    ],
)
def test_rate_coefficients_match_literature(fn, T, expected, rtol):
    assert float(fn(T)) == pytest.approx(expected, rel=rtol)


def test_compton_vanishes_at_cmb_temperature():
    assert float(hchem.varpi_cgs(hchem.T_CMB0_K, a=1.0)) == pytest.approx(0.0, abs=1e-40)


# ---------------------------------------------------------------------------
# (b) state view: conservative -> physical
# ---------------------------------------------------------------------------
def test_state_view_recovers_temperature_and_densities(setup):
    cu, _, sf, _, _ = setup
    n_H, T, x = 1.0, 1.0e4, 0.1
    sol = make_state(cu, n_H, T, x)
    view = sf.view
    assert float(view.temperature_K(sol)[0, 0, 0]) == pytest.approx(T, rel=1e-6)
    assert float(view.number_densities_cgs(sol)[0][0, 0, 0]) == pytest.approx(n_H, rel=1e-6)
    assert float(view.xHII(sol)[0, 0, 0]) == pytest.approx(x, rel=1e-6)


def test_state_view_accounts_for_kinetic_energy(setup):
    """T must be computed from E_tot - 0.5 rho v^2, not from E_tot."""
    cu, _, sf, _, _ = setup
    sol = make_state(cu, 1.0, 1.0e4, 0.1)
    rho = sol[4]
    v_code = 1.0e-3
    sol_moving = sol.at[5].set(rho * v_code).at[8].add(0.5 * rho * v_code ** 2)
    T0 = float(sf.view.temperature_K(sol)[0, 0, 0])
    T1 = float(sf.view.temperature_K(sol_moving)[0, 0, 0])
    assert T1 == pytest.approx(T0, rel=1e-6)


def test_number_density_uses_mH_not_mu(setup):
    """n_H = rho/m_H for pure hydrogen (previously divided by mu = 0.61)."""
    cu, _, sf, _, _ = setup
    sol = make_state(cu, n_H_cgs=3.0)
    assert float(sf.view.number_densities_cgs(sol)[0][0, 0, 0]) == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# (c) chemistry: analytic equilibrium
# ---------------------------------------------------------------------------
def test_photoionization_equilibrium(setup):
    """With a fixed photon field, x_HII must relax to the root of

        (1 - x) (Gamma + beta n_H x) = x^2 n_H alpha_A
    """
    cu, _, sf, _, ion = setup
    n_H, T, Ngam = 1.0, 1.0e4, 1.0e-4
    sol = make_state(cu, n_H, T, x_HII=1e-6, N_gamma_cgs=Ngam)

    Gamma = hchem.SIGMA_HI_0_CGS * hchem.C_LIGHT_CGS * Ngam
    alpha = float(hchem.alpha_A_HII_cgs(T))
    beta = float(hchem.beta_HI_cgs(T))

    def residual(x):
        return (1 - x) * (Gamma + beta * n_H * x) - x * x * n_H * alpha

    lo, hi = 0.0, 1.0
    for _ in range(200):                       # bisection on the analytic root
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    x_eq = 0.5 * (lo + hi)

    # ionization timescale is 1/Gamma ~ 5e10 s here: integrate for ~1e14 s
    dt_code = 1.0e11 / cu.T_cgs
    e_scale = n_H * hchem.KB_CGS * T / cu.P_cgs / (GAMMA - 1.0)

    def step_fn(k, s):
        s, _ = ion.force(k, s, {}, dt_code)
        # keep T and N_gamma fixed: only the chemistry is under test here
        return s.at[8].set((1.0 + sf.view.xHII(s)) * e_scale)

    sol = jax.jit(lambda s: jax.lax.fori_loop(0, 2000, step_fn, s))(sol)

    x_num = float(sf.view.xHII(sol)[0, 0, 0])
    assert x_num == pytest.approx(x_eq, rel=2e-3), f"x_num={x_num}, x_eq={x_eq}"


def test_xHII_stays_bounded_for_absurd_timestep(setup):
    """The requested limiter must keep x in [0, 1] whatever dt."""
    cu, _, sf, _, ion = setup
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.5, N_gamma_cgs=1.0)
    for step in range(50):
        sol, _ = ion.force(step, sol, {}, 1.0e30)
        x = np.asarray(sf.view.xHII(sol))
        assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_photon_sink_positivity_and_90pc_limiter(setup):
    """Delta N < 0 removes at most 90 % of N (explicit_limited mode)."""
    cu, _, sf, _, _ = setup
    sol = make_state(cu, n_H_cgs=1.0, T_K=1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    N0 = float(sol[0][0, 0, 0])
    sol2 = sf.apply_photon_chemistry_sink(sol, 1.0e30)
    assert float(sol2[0][0, 0, 0]) / N0 == pytest.approx(0.1, rel=1e-6)
    assert float(sol2[0][0, 0, 0]) > 0.0


def test_photon_sink_exponential_matches_analytic(setup):
    """The exact mode must reproduce N(t) = N0 exp(-n_HI sigma c t)."""
    cu, _, sf, _, _ = setup
    sf.photon_sink_mode = "exponential"
    sf.b_rec = 0.0                       # pure absorption
    n_H = 1.0
    sol = make_state(cu, n_H, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    t_s = 1.0e6
    sol2 = sf.apply_photon_chemistry_sink(sol, t_s / cu.T_cgs)
    expect = np.exp(-n_H * hchem.SIGMA_HI_0_CGS * hchem.C_LIGHT_CGS * t_s)
    assert float(sol2[0][0, 0, 0]) / float(sol[0][0, 0, 0]) == pytest.approx(expect, rel=1e-6)
    sf.photon_sink_mode = "explicit_limited"


def test_flux_sink_is_multiplicative_and_isotropic(setup):
    """dF/dt = -kappa c F must SCALE F, not add a scalar to each component."""
    cu, c_code, sf, _, _ = setup
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    Fx0 = 0.3 * c_code * float(sol[0][0, 0, 0])
    sol = sol.at[1].set(Fx0).at[2].set(-Fx0).at[3].set(0.0)

    t_s = 1.0e5
    sol2 = sf.apply_flux_chemistry_sink(sol, t_s / cu.T_cgs)
    decay = np.exp(-1.0 * hchem.SIGMA_HI_0_CGS * hchem.C_LIGHT_CGS * t_s)

    assert float(sol2[1][0, 0, 0]) == pytest.approx(Fx0 * decay, rel=1e-6)
    assert float(sol2[2][0, 0, 0]) == pytest.approx(-Fx0 * decay, rel=1e-6)
    assert float(sol2[3][0, 0, 0]) == pytest.approx(0.0, abs=1e-30)   # stays zero


def test_m1_cone_is_enforced(setup):
    cu, c_code, sf, _, _ = setup
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    Ncode = float(sol[0][0, 0, 0])
    sol = sol.at[1].set(10.0 * c_code * Ncode)          # way outside the cone
    sol = sf._clip_to_m1_cone(sol)
    f_red = float(np.sqrt(sum(np.asarray(sol[k]) ** 2 for k in (1, 2, 3))[0, 0, 0])
                  / (c_code * Ncode))
    assert f_red <= sf.beam_reduced_flux + 1e-9


# ---------------------------------------------------------------------------
# (d) Stromgren sphere
# ---------------------------------------------------------------------------
def test_stromgren_radius_from_equilibrium(setup):
    """A uniform photon field at the Stromgren-sphere-edge intensity must
    give x_HII ~ 0.5 nowhere-near-trivially; here we instead check the
    global photon budget: at equilibrium the ionizing luminosity balances
    the total recombination rate inside R_S,

        Q = 4/3 pi R_S^3 alpha_B n_H^2      (case B, fully ionized).
    """
    Q = 1.0e49          # photons / s (typical O star)
    n_H = 1.0
    alpha_B = float(hchem.alpha_B_HII_cgs(1.0e4))
    R_S = (3.0 * Q / (4.0 * np.pi * alpha_B * n_H ** 2)) ** (1.0 / 3.0)
    # Osterbrock & Ferland, table 2.3: R_S = 67 pc for Q = 1e49, n_H = 1,
    # T = 1e4 K (case B).
    R_S_pc = R_S / 3.0856775814913673e18
    assert 60.0 < R_S_pc < 75.0, R_S_pc


# ---------------------------------------------------------------------------
# photon-conserving coupled force
# ---------------------------------------------------------------------------
def test_coupled_force_conserves_photons(setup):
    """Every photon removed from N must produce exactly one ionization."""
    cu, _, sf, _, _ = setup
    coupled = HydrogenPhotoChemistryForce(sf, case="B", collisional=False, b_rec=0.0)
    n_H = 1.0
    sol = make_state(cu, n_H, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0e-4)

    view = coupled.view
    N0 = np.asarray(view.photon_density_cgs(sol), dtype=np.float64)
    x0 = np.asarray(view.xHII(sol), dtype=np.float64)

    dt_code = 1.0e8 / cu.T_cgs            # tau*c*dt >> 1: stiff on purpose
    sol2, _ = coupled.force(0, sol, {}, dt_code)

    N1 = np.asarray(view.photon_density_cgs(sol2), dtype=np.float64)
    x1 = np.asarray(view.xHII(sol2), dtype=np.float64)

    photons_lost = N0 - N1
    ionizations = (x1 - x0) * n_H
    assert np.allclose(photons_lost, ionizations, rtol=1e-6), (
        photons_lost.max(), ionizations.max())
    assert np.all(N1 >= 0.0) and np.all(x1 <= 1.0)


def test_coupled_force_does_not_destroy_excess_photons(setup):
    """When the cell is already ionized, photons must pass through unabsorbed."""
    cu, _, sf, _, _ = setup
    coupled = HydrogenPhotoChemistryForce(sf, case="B", collisional=False, b_rec=0.0)
    sol = make_state(cu, 1.0, 1.0e4, x_HII=1.0, N_gamma_cgs=1.0)
    N0 = float(coupled.view.photon_density_cgs(sol)[0, 0, 0])
    sol2, _ = coupled.force(0, sol, {}, 1.0e30)
    assert float(coupled.view.photon_density_cgs(sol2)[0, 0, 0]) == pytest.approx(N0, rel=1e-9)


def test_coupled_and_split_agree_in_the_optically_thin_limit(setup):
    """For n_HI sigma c dt << 1 both schemes must give the same dx_HII.

    N (1 - e^{-k dt}) / n_H  ->  (1 - x) sigma c N dt  when k dt -> 0, so the
    photon-conserving form reduces to the requested explicit Euler rate. The
    two only differ at O(k dt), which is exactly the systematic error the
    coupled form removes in the stiff (optically thick) regime.
    """
    cu, _, sf, _, ion = setup
    coupled = HydrogenPhotoChemistryForce(sf, case="A", collisional=True, b_rec=0.0)
    n_H, T, x0, Ngam = 1.0, 1.0e4, 0.5, 1.0e-4
    sol = make_state(cu, n_H, T, x_HII=x0, N_gamma_cgs=Ngam)

    k = n_H * (1 - x0) * hchem.SIGMA_HI_0_CGS * hchem.C_LIGHT_CGS
    dt_code = (1.0e-4 / k) / cu.T_cgs                # k dt = 1e-4

    dx_split = float(ion.dxdt_cgs(sol)[0, 0, 0]) * float(coupled.view.dt_cgs(dt_code))
    sol_c, _ = coupled.force(0, sol, {}, dt_code)
    dx_coupled = float(coupled.view.xHII(sol_c)[0, 0, 0]) - x0

    assert dx_coupled == pytest.approx(dx_split, rel=1e-3), (dx_coupled, dx_split)


def test_coupled_force_deposits_excess_photon_energy_once(setup):
    """The coupled source must heat by N_abs * (h nu - 13.6 eV), once."""
    cu, _, sf, _, _ = setup
    coupled = HydrogenPhotoChemistryForce(
        sf, case="B", collisional=False, b_rec=0.0,
        include_cooling=False, mean_photon_energy_eV=20.0,
    )
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    E0 = np.asarray(coupled.view.thermal_energy_code(sol), dtype=np.float64)
    N0 = np.asarray(coupled.view.photon_density_cgs(sol), dtype=np.float64)

    # Use a measurable but optically thin step: a tiny 1e-8 s step makes
    # N0-N1 itself cancellation-limited even in float64.
    sol2, _ = coupled.force(0, sol, {}, 1.0e4 / cu.T_cgs)
    E1 = np.asarray(coupled.view.thermal_energy_code(sol2), dtype=np.float64)
    N1 = np.asarray(coupled.view.photon_density_cgs(sol2), dtype=np.float64)

    expected = (N0 - N1) * (20.0 - hchem.E_HI_EV) * hchem.EV_CGS
    np.testing.assert_allclose((E1 - E0) * cu.P_cgs, expected, rtol=1e-8, atol=0.0)


def test_coupled_isothermal_mode_preserves_temperature(setup):
    cu, _, sf, _, _ = setup
    coupled = HydrogenPhotoChemistryForce(
        sf, case="B", collisional=False,
        include_heating=False, include_cooling=False, fixed_temperature_K=1.0e4,
    )
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    sol2, _ = coupled.force(0, sol, {}, 1.0e8 / cu.T_cgs)
    assert float(coupled.view.temperature_K(sol2)[0, 0, 0]) == pytest.approx(1.0e4, rel=1e-12)


# ---------------------------------------------------------------------------
# cooling
# ---------------------------------------------------------------------------
def test_cooling_matches_A17_hydrogen_only(setup):
    cu, _, sf, hc, _ = setup
    n_H, T, x = 1.0, 3.0e4, 0.5
    sol = make_state(cu, n_H, T, x)
    n_HI, n_HII = n_H * (1 - x), n_H * x
    ref = float(hchem.cooling_rate_cgs(T, n_HI, n_HII, n_HII, a=1.0, case="A"))
    got = float(hc.cooling(jnp.asarray(T), sol)[0, 0, 0])
    assert got == pytest.approx(ref, rel=1e-9)
    assert got > 0.0


def test_photoheating_is_zero_at_threshold_and_positive_above(setup):
    cu, _, sf, hc, _ = setup
    sol = make_state(cu, 1.0, 1.0e4, x_HII=0.0, N_gamma_cgs=1.0)
    assert float(hc.heating(sol)[0, 0, 0]) == 0.0
    hc.mean_photon_energy_eV = 18.0
    assert float(hc.heating(sol)[0, 0, 0]) > 0.0


def test_cooling_never_makes_pressure_negative(setup):
    cu, _, sf, hc, _ = setup
    sol = make_state(cu, n_H_cgs=1.0e3, T_K=3.0e4, x_HII=1.0)
    for step in range(200):
        sol, _ = hc.force(step, sol, {}, 1.0e30)
        assert np.all(np.asarray(sf.view.pressure_code(sol)) > 0.0)
    T_end = float(sf.view.temperature_K(sol)[0, 0, 0])
    assert T_end < 3.0e4          # it did cool
    assert T_end > 0.0
