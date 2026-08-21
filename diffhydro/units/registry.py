"""Small unit parser/registry for boundary-layer unit conversions."""

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
        "K": (1.0, "temperature"),
    }

    _DEFAULT_CGS_UNITS = {
        "length": "cm",
        "mass": "g",
        "time": "s",
        "velocity": "cm/s",
        "density": "g/cm^3",
        "pressure": "dyne/cm^2",
        "energy_density": "erg/cm^3",
        "radiation_flux": "erg/s/cm^2",
        "temperature": "K",
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
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(
                f"Unit '{unit}' has dimension '{dim}', expected '{expected_dim}'."
            )
        return ParsedQuantity(value=value, unit=unit, dimension=dim, cgs_value=value * factor)

    def unit_factor_to_cgs(self, unit: str, expected_dim: Optional[str] = None) -> float:
        norm = self._normalize_unit(unit)
        factor, dim = self._unit_info(norm)
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(
                f"Unit '{norm}' has dimension '{dim}', expected '{expected_dim}'."
            )
        return factor

    def unit_dimension(self, unit: str) -> str:
        norm = self._normalize_unit(unit)
        _, dim = self._unit_info(norm)
        return dim

    def default_cgs_unit(self, dim: str) -> str:
        try:
            return self._DEFAULT_CGS_UNITS[dim]
        except KeyError as exc:
            raise ValueError(f"Unknown dimension '{dim}'.") from exc

    def units_for_dimension(self, dim: str) -> list[tuple[str, float]]:
        """Return [(unit_name, cgs_factor), ...] for every known unit of
        the given dimension, sorted by increasing cgs factor.

        Used by callers that need to auto-select a human-readable display
        unit (e.g. picking 'km' instead of 'cm' for a large box, or 'cm'
        instead of 'km' for a small one) without hardcoding a unit list
        that could drift out of sync with _UNIT_TABLE.
        """
        entries = [
            (unit, factor)
            for unit, (factor, d) in self._UNIT_TABLE.items()
            if d == dim
        ]
        return sorted(entries, key=lambda item: item[1])

    def _unit_info(self, unit: str) -> tuple[float, str]:
        try:
            return self._UNIT_TABLE[unit]
        except KeyError as exc:
            allowed = ", ".join(sorted(self._UNIT_TABLE))
            raise ValueError(f"Unsupported unit '{unit}'. Known units: {allowed}") from exc

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        return "".join(unit.strip().split())
