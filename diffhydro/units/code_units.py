"""Code-unit definition and derived physical scales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import UnitParser


@dataclass(frozen=True)
class CodeUnits:
    """Single source of truth for code-unit scales."""

    L_cgs: float
    M_cgs: float
    V_cgs: float
    gamma: float = 5.0 / 3.0
    mu: float = 0.6
    kB_cgs: float = 1.380649e-16
    mH_cgs: float = 1.6735575e-24

    @property
    def T_cgs(self) -> float:
        return self.L_cgs / self.V_cgs

    @property
    def rho_cgs(self) -> float:
        return self.M_cgs / (self.L_cgs**3)

    @property
    def P_cgs(self) -> float:
        return self.rho_cgs * (self.V_cgs**2)

    @property
    def Eden_cgs(self) -> float:
        return self.P_cgs

    @property
    def Temp_cgs(self) -> float:
        return self.mu * self.mH_cgs * (self.V_cgs**2) / self.kB_cgs

    def scale(self, dim: str) -> float:
        scales = {
            "length": self.L_cgs,
            "mass": self.M_cgs,
            "time": self.T_cgs,
            "velocity": self.V_cgs,
            "density": self.rho_cgs,
            "pressure": self.P_cgs,
            "energy_density": self.Eden_cgs,
            "temperature": self.Temp_cgs,
        }
        try:
            return scales[dim]
        except KeyError as exc:
            raise ValueError(f"Unknown dimension '{dim}'.") from exc

    @classmethod
    def from_config(
        cls,
        code_units_cfg: Mapping[str, Any],
        thermo_cfg: Mapping[str, Any] | None = None,
        parser: UnitParser | None = None,
    ) -> "CodeUnits":
        unit_parser = parser or UnitParser()
        thermo_cfg = thermo_cfg or {}

        L_cgs = _parse_base_scale(code_units_cfg["length"], "length", unit_parser)
        M_cgs = _parse_base_scale(code_units_cfg["mass"], "mass", unit_parser)
        V_cgs = _parse_base_scale(code_units_cfg["velocity"], "velocity", unit_parser)

        return cls(
            L_cgs=L_cgs,
            M_cgs=M_cgs,
            V_cgs=V_cgs,
            gamma=float(thermo_cfg.get("gamma", 5.0 / 3.0)),
            mu=float(thermo_cfg.get("mu", 0.6)),
        )


def _parse_base_scale(value: Any, expected_dim: str, parser: UnitParser) -> float:
    if isinstance(value, str):
        if value.strip().lower().startswith("code:"):
            return float(value.split(":", 1)[1].strip())
        parsed = parser.parse(value, expected_dim=expected_dim)
        return parsed.cgs_value
    return float(value)

