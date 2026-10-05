"""
Example: How to use units in the radiative transfer module.

This example shows:
1. Creating CodeUnits from config
2. Converting physical inputs to code units
3. Using radiative fields with proper dimensions
4. Converting results back to physical units
"""

import jax.numpy as jnp
from diffhydro.units import (
    CodeUnits,
    UnitParser,
    FIELD_DIMS,
    to_code,
    from_code,
    to_code_fields,
    from_code_fields,
)
from diffhydro.physics.radiative_transfer import StellarRadiationForce


def example_setup_with_units():
    """Setup with units: define code units from physical scales."""

    # Step 1: Define code units from physical scales
    code_units_cfg = {
        "length": "1e22 cm",      # ~1 kpc in cm
        "mass": "1e11 Msun",      # ~100 billion solar masses
        "velocity": "1e5 cm/s",   # 100 km/s
    }

    thermo_cfg = {
        "gamma": 5.0 / 3.0,  # diatomic gas
        "mu": 0.6,           # mean molecular weight
    }

    cu = CodeUnits.from_config(code_units_cfg, thermo_cfg)
    parser = UnitParser()

    print("=== Code Units ===")
    print(f"Length scale: {cu.L_cgs:.2e} cm")
    print(f"Mass scale: {cu.M_cgs:.2e} g")
    print(f"Velocity scale: {cu.V_cgs:.2e} cm/s")
    print(f"Time scale: {cu.T_cgs:.2e} s")
    print(f"Density scale: {cu.rho_cgs:.2e} g/cm³")
    print(f"Pressure scale: {cu.P_cgs:.2e} dyne/cm²")
    print(f"Energy density scale: {cu.Eden_cgs:.2e} erg/cm³")
    print(f"Radiation flux scale: {cu.RadFlux_cgs:.2e} erg/s/cm²")

    return cu, parser


def example_input_conversion():
    """Convert physical input parameters to code units."""

    cu, parser = example_setup_with_units()

    # Step 2: Convert physical parameters to code units
    print("\n=== Converting Input Parameters ===")

    # Initial density
    rho_phys = "1e-21 g/cm^3"  # Physical value
    rho_code = to_code(rho_phys, "density", cu, parser)
    print(f"Density: {rho_phys} → {rho_code:.4e} code units")

    # Stellar mass
    stellar_mass_phys = "10 Msun"
    stellar_mass_code = to_code(stellar_mass_phys, "mass", cu, parser)
    print(f"Stellar mass: {stellar_mass_phys} → {stellar_mass_code:.4e} code units")

    # Radiation flux
    flux_phys = "1e5 erg/s/cm^2"
    flux_code = to_code(flux_phys, "radiation_flux", cu, parser)
    print(f"Radiation flux: {flux_phys} → {flux_code:.4e} code units")


def example_radiative_force_with_units():
    """Setup StellarRadiationForce with unit support."""

    cu, parser = example_setup_with_units()

    # Create a mock equation manager with necessary attributes
    class MockEq:
        light_speed = 1.0  # Code units: c = 1 in natural units
        mesh_shape = (100, 100, 100)
        eps = 1e-20

    eq = MockEq()

    print("\n=== Setting up StellarRadiationForce ===")

    # Configure stellar radiation in physical units
    stellar_config = {
        "escape_fraction": 0.1,
        "dx": 1.0,  # Code units (already converted from physical)
        "stromgren_rate": to_code("1e50 erg/s", "energy_density", cu, parser),
        "gaussian_star": True,
        "injection_geometry": "3D",
        "injection_momentum": True,
        "beam_momentum_scaling": "physical",
    }

    srf = StellarRadiationForce(
        eq=eq,
        **stellar_config
    )

    # Store CodeUnits for conversions
    srf.cu = cu
    srf.parser = parser

    print(f"Escape fraction: {srf.escape_fraction}")
    print(f"Injection mode: {srf.injection_mode}")
    print(f"CodeUnits attached: {srf.cu is not None}")

    return srf, cu, parser


def example_convert_output_to_physical():
    """Convert simulation results back to physical units."""

    cu, parser = example_setup_with_units()

    print("\n=== Converting Output to Physical Units ===")

    # Simulate some code-unit results
    E_gamma_code = jnp.array([1e-8, 5e-8, 1e-7])  # Code units
    Fx_code = jnp.array([1e-10, 2e-10, 3e-10])    # Code units

    # Convert back to physical units
    E_physical = from_code(E_gamma_code, "energy_density", cu, "erg/cm^3", parser)
    Fx_physical = from_code(Fx_code, "radiation_flux", cu, "erg/s/cm^2", parser)

    print(f"E_gamma (code): {E_gamma_code}")
    print(f"E_gamma (physical): {E_physical.value} {E_physical.unit}")

    print(f"\nFx (code): {Fx_code}")
    print(f"Fx (physical): {Fx_physical.value} {Fx_physical.unit}")


def example_batch_field_conversion():
    """Convert multiple fields at once."""

    cu, parser = example_setup_with_units()

    print("\n=== Batch Field Conversion ===")

    # Physical fields
    physical_fields = {
        "rho": "1e-20 g/cm^3",
        "vx": "5e4 cm/s",
        "p": "1e-9 dyne/cm^2",
    }

    # Convert to code units
    code_fields = to_code_fields(physical_fields, cu, FIELD_DIMS, parser)
    print("Physical → Code:")
    for name, value in code_fields.items():
        print(f"  {name}: {value:.4e}")

    # Simulate results in code units
    results_code = {
        "rho": code_fields["rho"],
        "vx": code_fields["vx"],
        "p": code_fields["p"],
        "E_gamma": 1e-8,  # Radiation energy density
        "Fx_gamma": 1e-10,  # Radiation flux
    }

    # Convert back to physical
    display_units = {
        "density": "g/cm^3",
        "velocity": "cm/s",
        "pressure": "dyne/cm^2",
        "energy_density": "erg/cm^3",
        "radiation_flux": "erg/s/cm^2",
    }

    results_physical = from_code_fields(results_code, cu, display_units, FIELD_DIMS, parser)
    print("\nCode → Physical:")
    for name, quantity in results_physical.items():
        print(f"  {name}: {quantity.value:.4e} {quantity.unit}")


if __name__ == "__main__":
    print("=" * 60)
    print("RADIATIVE TRANSFER WITH UNITS - COMPLETE EXAMPLE")
    print("=" * 60)

    example_setup_with_units()
    example_input_conversion()
    example_radiative_force_with_units()
    example_convert_output_to_physical()
    example_batch_field_conversion()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. In StellarRadiationForce.__init__, accept 'cu' parameter")
    print("2. Store self.cu = cu and self.parser = parser")
    print("3. Use cu.RadFlux_cgs for converting radiation flux values")
    print("4. Use from_code_fields() when outputting results")
