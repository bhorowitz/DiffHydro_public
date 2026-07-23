"""Unit parsing and registry helpers for code/CGS conversions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional


_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\n]+?)\s*$"
)


@dataclass(frozen=True)
class ParsedQuantity:
    value: float
    unit: str
    dimension: str
    cgs_value: float


@dataclass(frozen=True)
class Quantity:
    value: Any
    unit: str
    dimension: str


class UnitParser:
    """Parses user-facing quantities and converts between known units."""

    _DIM_ALIASES = {
        "radiation_energy_density": "energy_density",
        "photon_density": "photon_number_density",
        "photon_number_density": "photon_number_density",
        "light_speed": "velocity",
    }

    _UNIT_TABLE = {
        "cm": (1.0, "length"),
        "m": (100.0, "length"),
        "km": (1.0e5, "length"),
        "pc": (3.0856775814913673e18, "length"),
        "kpc": (3.0856775814913673e21, "length"),
        "g": (1.0, "mass"),
        "kg": (1.0e3, "mass"),
        "Msun": (1.98847e33, "mass"),
        "M_sun": (1.98847e33, "mass"),
        "s": (1.0, "time"),
        "cm/s": (1.0, "velocity"),
        "m/s": (100.0, "velocity"),
        "km/s": (1.0e5, "velocity"),
        "g/cm^3": (1.0, "density"),
        "kg/m^3": (1.0e-3, "density"),
        "dyne/cm^2": (1.0, "pressure"),
        "erg/cm^3": (1.0, "energy_density"),
        "erg/s/cm^2": (1.0, "radiation_flux"),
        "erg/s": (1.0, "power"),
        "erg": (1.0, "energy"),
        "J": (1.0e7, "energy"),
        "Pa": (10.0, "pressure"),
        "photons/cm^2": (1.0, "photon_surface_density"),
        "cm^-2": (1.0, "photon_surface_density"),
        "1/cm^2": (1.0, "photon_surface_density"),
        "photons/cm^3": (1.0, "photon_number_density"),
        "cm^-3": (1.0, "photon_number_density"),
        "1/cm^3": (1.0, "photon_number_density"),
        "photons/s/cm^2": (1.0, "photon_flux"),
        "photons/cm^2/s": (1.0, "photon_flux"),
        "cm^-2/s": (1.0, "photon_flux"),
        "1/cm^2/s": (1.0, "photon_flux"),
        "photons/s": (1.0, "photon_rate"),
        "s^-1": (1.0, "photon_rate"),
        "1/s": (1.0, "photon_rate"),
        "c": (3.0e10, "light_speed"),
        "3e10 cm/s": (3.0e10, "light_speed"),
        "K": (1.0, "temperature"),
    }

    _DEFAULT_CGS_UNITS = {
        "length": "cm",
        "mass": "g",
        "time": "s",
        "velocity": "cm/s",
        "energy": "erg",
        "power": "erg/s",
        "density": "g/cm^3",
        "pressure": "dyne/cm^2",
        "energy_density": "erg/cm^3",
        "radiation_flux": "erg/s/cm^2",
        "photon_surface_density": "photons/cm^2",
        "photon_number_density": "photons/cm^3",
        "photon_flux": "photons/s/cm^2",
        "photon_rate": "photons/s",
        "temperature": "K",
        "light_speed": "c",
    }

    def parse(self, text: str, expected_dim: Optional[str] = None) -> ParsedQuantity:
        match = _NUMBER_RE.match(text)
        if match is None:
            raise ValueError(
                f"Could not parse quantity '{text}'. Expected format like '1e4 K'."
            )

        value = float(match.group(1))
        unit = self._normalize_unit(match.group(2))
        factor, dim = self._unit_info(unit)
        expected_dim = self._normalize_dimension(expected_dim)
        if not self._dimension_matches(dim, expected_dim):
            raise ValueError(
                f"Unit '{unit}' has dimension '{dim}', expected '{expected_dim}'."
            )
        return ParsedQuantity(value=value, unit=unit, dimension=dim, cgs_value=value * factor)

    def unit_factor_to_cgs(self, unit: str, expected_dim: Optional[str] = None) -> float:
        norm = self._normalize_unit(unit)
        factor, dim = self._unit_info(norm)
        expected_dim = self._normalize_dimension(expected_dim)
        if not self._dimension_matches(dim, expected_dim):
            raise ValueError(
                f"Unit '{norm}' has dimension '{dim}', expected '{expected_dim}'."
            )
        return factor

    def unit_dimension(self, unit: str) -> str:
        norm = self._normalize_unit(unit)
        _, dim = self._unit_info(norm)
        return dim

    def default_cgs_unit(self, dim: str) -> str:
        dim = self._normalize_dimension(dim)
        try:
            return self._DEFAULT_CGS_UNITS[dim]
        except KeyError as exc:
            raise ValueError(f"Unknown dimension '{dim}'.") from exc

    def _unit_info(self, unit: str) -> tuple[float, str]:
        try:
            return self._UNIT_TABLE[unit]
        except KeyError as exc:
            allowed = ", ".join(sorted(self._UNIT_TABLE))
            raise ValueError(f"Unsupported unit '{unit}'. Known units: {allowed}") from exc

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        return "".join(unit.strip().split())

    def _normalize_dimension(self, dim: Optional[str]) -> Optional[str]:
        if dim is None:
            return None
        return self._DIM_ALIASES.get(dim, dim)

    @staticmethod
    def _dimension_matches(actual: str, expected: Optional[str]) -> bool:
        if expected is None:
            return True
        if actual == expected:
            return True
        if {actual, expected} <= {"velocity", "light_speed"}:
            return True
        return False

