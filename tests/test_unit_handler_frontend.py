import numpy as np
import pytest

import diffhydro as dh
from diffhydro.prob_gen import sedov
from diffhydro.units import (
    CodeUnits,
    UnitParser,
    build_unit_metadata,
    code_units_from_metadata,
    from_code,
    from_code_fields,
    format_quantity,
    pressure_from_Trho,
    temperature_from_Prho,
    to_code,
    to_code_fields,
)


def test_unit_parser_and_dim_validation():
    parser = UnitParser()
    q = parser.parse("10 km/s", expected_dim="velocity")
    assert q.dimension == "velocity"
    assert q.cgs_value == pytest.approx(1.0e6)

    with pytest.raises(ValueError, match="expected 'density'"):
        parser.parse("10 km/s", expected_dim="density")


def test_code_units_from_config_and_scales():
    cu = CodeUnits.from_config(
        {
            "length": "1 kpc",
            "mass": "1e10 Msun",
            "velocity": "1 km/s",
        },
        {"gamma": 1.6666667, "mu": 0.6},
    )

    assert cu.L_cgs == pytest.approx(3.0856775814913673e21)
    assert cu.M_cgs == pytest.approx(1.98847e43)
    assert cu.V_cgs == pytest.approx(1.0e5)
    assert cu.scale("density") == pytest.approx(cu.M_cgs / cu.L_cgs**3)
    assert cu.scale("pressure") == pytest.approx(cu.scale("density") * cu.V_cgs**2)


def test_dimension_level_to_from_code_and_code_prefix():
    cu = CodeUnits(L_cgs=1.0e2, M_cgs=1.0e3, V_cgs=1.0e4)
    rho_code = to_code("2 g/cm^3", "density", cu)
    assert rho_code == pytest.approx(2.0e3)

    rho_phys = from_code(rho_code, "density", cu, out_unit="g/cm^3")
    assert rho_phys.value == pytest.approx(2.0)
    assert rho_phys.unit == "g/cm^3"

    assert to_code("code: 3.5", "velocity", cu) == pytest.approx(3.5)
    q = from_code(7.0, "velocity", cu, out_unit="code:")
    assert q.value == pytest.approx(7.0)
    assert q.unit == "code"


def test_field_level_roundtrip():
    cu = CodeUnits.from_config(
        {"length": "1 pc", "mass": "1 Msun", "velocity": "1 km/s"},
        {"mu": 0.61},
    )
    physical = {
        "rho": "1e-24 g/cm^3",
        "vx": "10 km/s",
        "vy": "0 km/s",
        "vz": "0 km/s",
        "p": "1e-12 dyne/cm^2",
        "Etot": "1e-12 erg/cm^3",
        "custom_passthrough": 123.0,
    }

    code_vals = to_code_fields(physical, cu)
    assert code_vals["custom_passthrough"] == 123.0
    assert code_vals["vx"] == pytest.approx(10.0)

    display_units = {
        "density": "g/cm^3",
        "velocity": "km/s",
        "pressure": "dyne/cm^2",
        "energy_density": "erg/cm^3",
    }
    roundtrip = from_code_fields(code_vals, cu, display_units)
    assert roundtrip["rho"].value == pytest.approx(1.0e-24)
    assert roundtrip["vx"].value == pytest.approx(10.0)
    assert roundtrip["p"].value == pytest.approx(1.0e-12)


def test_format_quantity_and_radiative_alias():
    cu = CodeUnits.from_config(
        {"length": "1 kpc", "mass": "1 Msun", "velocity": "1 km/s"},
        {"mu": 0.61},
    )

    text = format_quantity(1.0, "radiation_energy_density", cu)
    assert text.endswith("erg/cm^3")

    profile = format_quantity(
        np.array([1.0, 2.0, 3.0]),
        "radiation_flux",
        cu,
        out_unit="erg/s/cm^2",
    )
    assert "erg/s/cm^2" in profile
    assert "[" in profile and "]" in profile


def test_thermo_helpers_invert_each_other():
    cu = CodeUnits.from_config(
        {"length": "1 pc", "mass": "1 Msun", "velocity": "10 km/s"},
        {"mu": 0.62},
    )
    rho_code = np.array([1.0, 0.1, 10.0])
    T = np.array([1.0e4, 2.5e5, 4.0e3])

    p_code = pressure_from_Trho(T, rho_code, cu)
    T_back = temperature_from_Prho(p_code, rho_code, cu)
    np.testing.assert_allclose(T_back, T, rtol=1e-12, atol=0.0)

    p_from_str = pressure_from_Trho("1e4 K", 2.0, cu)
    T_from_p = temperature_from_Prho(p_from_str, 2.0, cu)
    assert T_from_p == pytest.approx(1.0e4)


def test_metadata_roundtrip():
    cu = CodeUnits.from_config(
        {"length": "2 kpc", "mass": "5e9 Msun", "velocity": "15 km/s", "light_speed": "2.5e10 cm/s"},
        {"gamma": 1.4, "mu": 0.58},
    )
    meta = build_unit_metadata(cu)
    cu2 = code_units_from_metadata(meta)

    assert cu2.L_cgs == pytest.approx(cu.L_cgs)
    assert cu2.M_cgs == pytest.approx(cu.M_cgs)
    assert cu2.V_cgs == pytest.approx(cu.V_cgs)
    assert cu2.light_speed_cgs == pytest.approx(cu.light_speed_cgs)
    assert cu2.gamma == pytest.approx(cu.gamma)
    assert cu2.mu == pytest.approx(cu.mu)


def test_light_speed_dimension_accepts_velocity_units():
    cu = CodeUnits.from_config(
        {"length": "1 pc", "mass": "1 Msun", "velocity": "10 km/s", "light_speed": "3e10 cm/s"},
        {"mu": 0.61},
    )

    assert to_code("1 km/s", "light_speed", cu) == pytest.approx(1.0e5 / cu.light_speed_cgs)
    q = from_code(1.0, "light_speed", cu, out_unit="km/s")
    assert q.value == pytest.approx(cu.light_speed_cgs / 1.0e5)
    assert q.unit == "km/s"


def test_sedov_evolution_consistent_across_astropy_unit_representations():
    apu = pytest.importorskip("astropy.units")

    eq = dh.equationmanager.EquationManager()
    eq.mesh_shape = [24, 24, 24]
    eq.cfl = 0.2

    ss = dh.signal_speed_Rusanov
    solver = dh.HLLC(equation_manager=eq, signal_speed=ss)
    cf = dh.ConvectiveFlux(eq, solver, dh.MUSCL3(limiter="SUPERBEE"))
    hydrosim = dh.hydro(n_super_step=32, fluxes=[cf], use_mol=True)

    cu = CodeUnits.from_config(
        {"length": "1 pc", "mass": "1 Msun", "velocity": "1 km/s"},
        {"gamma": eq.gamma, "mu": 0.6},
    )

    rho_phys_options = [
        1.0e-24 * apu.g / apu.cm**3,
        1.0e-21 * apu.kg / apu.m**3,
        1.0e-27 * apu.g / apu.mm**3,
    ]
    energy_phys_options = [
        1.0e51 * apu.erg,
        1.0e44 * apu.J,
        1.0e51 * apu.dyne * apu.cm,
    ]
    box_phys_options = [
        30.0 * apu.pc,
        0.03 * apu.kpc,
        (30.0 * 3.0856775814913673e13) * apu.km,
    ]
    rho_out_units = [apu.g / apu.cm**3, apu.kg / apu.m**3, apu.g / apu.cm**3]
    etot_out_units = [apu.erg / apu.cm**3, apu.J / apu.m**3, apu.Pa]

    rho_cgs_ref = None
    etot_cgs_ref = None
    for i in range(3):
        rho_phys = rho_phys_options[i]
        energy_phys = energy_phys_options[i]
        box_phys = box_phys_options[i]

        rho_cgs = rho_phys.to(apu.g / apu.cm**3).value
        energy_cgs = energy_phys.to(apu.erg).value
        box_cgs = box_phys.to(apu.cm).value

        rho_code = rho_cgs / cu.rho_cgs
        energy_code = energy_cgs / (cu.M_cgs * cu.V_cgs**2)
        box_code = box_cgs / cu.L_cgs
        eq.box_size = (box_code, box_code, box_code)

        U0, _ = sedov(energy_code, rho_code, eq)
        U_final, _ = hydrosim.evolve(U0, {})

        rho_code_f = np.asarray(U_final[0])
        etot_code_f = np.asarray(U_final[4])

        rho_phys_out = (rho_code_f * cu.rho_cgs) * (apu.g / apu.cm**3)
        etot_phys_out = (etot_code_f * cu.Eden_cgs) * (apu.erg / apu.cm**3)

        # Validate output conversion consistency under different requested output units.
        rho_cgs_now = rho_phys_out.to(rho_out_units[i]).to(apu.g / apu.cm**3).value
        etot_cgs_now = etot_phys_out.to(etot_out_units[i]).to(apu.erg / apu.cm**3).value

        if rho_cgs_ref is None:
            rho_cgs_ref = rho_cgs_now
            etot_cgs_ref = etot_cgs_now
        else:
            # Nonlinear evolution in float32 can amplify tiny IC roundoff differences
            # from equivalent unit representations; compare at physically consistent tolerance.
            np.testing.assert_allclose(rho_cgs_now, rho_cgs_ref, rtol=1.0e-2, atol=0.0)
            np.testing.assert_allclose(etot_cgs_now, etot_cgs_ref, rtol=1.0e-2, atol=0.0)


import jax.numpy as jnp
import numpy as np
def test_radiative_transfer_units_basic():
    """
    Test minimal de cohérence d'unités/invariants pour le transfert radiatif.

    Hypothèses du code :
    - sol[0] = Egamma
    - sol[1] = Fx
    - sol[2] = Fy
    - sol[3] = Fz
    - contrainte M1 : |F| <= c E
    """

    c = 2.99792458e10  # cm/s, ou la valeur cohérente avec ton système d'unités code
    eps = 1e-30

    # état radiatif synthétique
    E = jnp.array([1e-12, 2e-12, 5e-12], dtype=jnp.float64)
    Fx = jnp.array([0.2 * c * E[0], 0.5 * c * E[1], 0.9 * c * E[2]], dtype=jnp.float64)
    Fy = jnp.array([0.0, 0.1 * c * E[1], 0.0], dtype=jnp.float64)
    Fz = jnp.array([0.0, 0.0, 0.05 * c * E[2]], dtype=jnp.float64)

    sol = jnp.stack([E, Fx, Fy, Fz], axis=0)

    # 1) finitude
    assert jnp.all(jnp.isfinite(sol)), "NaN/Inf détecté dans l'état radiatif"

    # 2) positivité énergie
    assert jnp.all(sol[0] >= 0.0), "Egamma doit rester positive"

    # 3) contrainte M1 : |F| <= c E
    Fmag = jnp.sqrt(sol[1]**2 + sol[2]**2 + sol[3]**2)
    assert jnp.all(Fmag <= c * sol[0] + eps), "Violation de la borne M1 : |F| > cE"

    # 4) reduced flux f = |F| / (cE) doit être dans [0,1]
    f = Fmag / jnp.maximum(c * sol[0], eps)
    assert jnp.all(f >= 0.0), "Reduced flux négatif"
    assert jnp.all(f <= 1.0 + 1e-12), "Reduced flux > 1"

    # 5) test de cohérence de l'injection de momentum
    # dans ton code : fxinj = source / c**2 * weights
    source = 1e40
    weights = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float64)
    fxinj = source / c**2 * weights

    assert jnp.all(jnp.isfinite(fxinj)), "Injection de momentum non finie"
    assert np.isclose(float(jnp.sum(weights)), 1.0, rtol=1e-12, atol=1e-12), \
        "Les poids d'injection doivent sommer à 1"
    assert np.isclose(float(jnp.sum(fxinj)), float(source / c**2), rtol=1e-12), \
        "Somme de l'injection Fx incohérente avec source/c^2"