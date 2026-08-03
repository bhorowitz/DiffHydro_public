"""
Modified StellarRadiationForce class with unit support.

This shows the pattern to integrate units into your radiative transfer module.
Copy these modifications into diffhydro/physics/radiative_transfer.py
"""

from diffhydro.units import CodeUnits, UnitParser


class StellarRadiationForceWithUnits:
    """
    Radiative source term from stellar populations - WITH UNIT SUPPORT.

    Key additions:
    - Accept CodeUnits in __init__
    - Use cu for converting physical parameters
    - Track both code and physical values separately
    """

    def __init__(
        self,
        escape_fraction=0.1,
        stellar_spectrum_func=None,
        dx=1.0,
        injection_mode="physical",
        stromgren_rate=1e-7,
        gaussian_star=True,
        injection_geometry="3D",
        injection_momentum=False,
        momentum_only=False,
        eq=None,
        debug=False,
        one_injection=False,
        beam_axis=0,
        beam_sign=+1,
        beam_length_cells=8,
        beam_sigma=3.0,
        beam_reduced_flux=0.95,
        beam_momentum_scaling="physical",
        # ─── NEW: Unit support ───
        cu: CodeUnits = None,
        parser: UnitParser = None,
    ):
        self.escape_fraction = escape_fraction
        self.stellar_spectrum_func = stellar_spectrum_func
        self.dx = dx
        self.injection_mode = injection_mode
        self.stromgren_rate = stromgren_rate
        self.gaussian_star = gaussian_star
        self.injection_geometry = injection_geometry
        self.injection_momentum = injection_momentum
        self.momentum_only = momentum_only
        self.debug = debug
        self.light_speed = eq.light_speed if eq is not None else 1.0
        self.mesh_shape = eq.mesh_shape if eq is not None else (100, 100, 100)
        self.one_injection = one_injection
        self.beam_axis = beam_axis
        self.beam_sign = beam_sign
        self.beam_length_cells = beam_length_cells
        self.beam_sigma = beam_sigma
        self.beam_reduced_flux = beam_reduced_flux
        self.beam_momentum_scaling = beam_momentum_scaling
        self.sol = None
        self.eq = eq

        # ─── NEW: Store CodeUnits for conversions ───
        self.cu = cu
        self.parser = parser or UnitParser()

    # ─── NEW: Helper methods for unit conversion ───

    def set_code_units(self, cu: CodeUnits, parser: UnitParser = None):
        """Set CodeUnits after initialization (if needed)."""
        self.cu = cu
        self.parser = parser or UnitParser()

    def stromgren_rate_to_code(self, value_str: str) -> float:
        """
        Convert physical stromgren rate to code units.

        Example:
            "1e50 erg/s" → code units
        """
        if self.cu is None:
            raise ValueError("CodeUnits not set. Call set_code_units() first.")

        from diffhydro.units import to_code
        return to_code(value_str, "energy_density", self.cu, self.parser)

    def radiation_flux_to_physical(self, flux_code: float, unit: str = "erg/s/cm^2") -> float:
        """
        Convert code-unit radiation flux to physical units.

        Example:
            1e-8 (code) → physical value in erg/s/cm²
        """
        if self.cu is None:
            raise ValueError("CodeUnits not set. Call set_code_units() first.")

        from diffhydro.units import from_code
        result = from_code(flux_code, "radiation_flux", self.cu, unit, self.parser)
        return result.value

    # ─── Rest of the class methods remain the same ───
    # but now have access to self.cu for any calculations


# ─── USAGE EXAMPLES ───

def example_usage():
    """Show how to use the modified class."""

    from diffhydro.units import CodeUnits

    # 1. Setup code units
    cu = CodeUnits.from_config(
        {
            "length": "1e22 cm",
            "mass": "1e11 Msun",
            "velocity": "1e5 cm/s",
        },
        {"gamma": 5.0 / 3.0, "mu": 0.6},
    )

    # 2. Mock equation manager
    class MockEq:
        light_speed = 1.0
        mesh_shape = (100, 100, 100)
        eps = 1e-20

    # 3. Create radiation force WITH units
    srf = StellarRadiationForceWithUnits(
        escape_fraction=0.1,
        stromgren_rate=1e-7,
        gaussian_star=True,
        eq=MockEq(),
        cu=cu,  # ← Pass CodeUnits here
    )

    # 4. Now you can use unit-aware methods
    print(f"CodeUnits set: {srf.cu is not None}")
    print(f"Radiation flux scale: {srf.cu.RadFlux_cgs:.2e} erg/s/cm²")

    # 5. Convert physical stromgren rate to code units
    # If you want to accept a physical parameter:
    # stromgren_phys = "1e50 erg/s"
    # stromgren_code = srf.stromgren_rate_to_code(stromgren_phys)

    # 6. Convert output back to physical
    # flux_code = 1e-8  # From simulation
    # flux_phys = srf.radiation_flux_to_physical(flux_code)
    # print(f"Flux: {flux_phys:.2e} erg/s/cm²")


if __name__ == "__main__":
    example_usage()
