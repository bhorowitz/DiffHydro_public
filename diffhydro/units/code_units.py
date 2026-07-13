"""Code-unit definition and derived physical scales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import UnitParser


@dataclass(frozen=True)
class CodeUnits:
    """Single source of truth for code-unit scales."""

    L_cgs: float                 # hydro / gas length scale
    M_cgs: float                 # hydro / gas mass scale
    V_cgs: float                 # hydro / gas velocity scale
    c_rt_cgs: float = 3.0e10     # radiative transfer light speed (physical or reduced) cm.s^-1
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

    @property
    def RadFlux_cgs(self) -> float:
        return self.PhotonFlux_cgs

    @property
    def RadEnergy_cgs(self) -> float:
        return self.PhotonNumberDensity_cgs

    @property
    def PhotonSurfaceDensity_cgs(self) -> float:
        return 1.0 / (self.L_cgs**2)

    @property
    def PhotonNumberDensity_cgs(self) -> float:
        return 1.0 / (self.L_cgs**3)

    @property
    def PhotonNumber_cgs(self) -> float:
        return 1.0

    @property
    def PhotonRate_cgs(self) -> float:
        return 1.0 / self.T_cgs

    @property
    def PhotonFlux_cgs(self) -> float:
        return 1.0 / (self.L_cgs**2 * self.T_cgs)

    @property
    def light_speed_cgs(self) -> float:
        """RT light speed in cgs: can be physical c or reduced c."""
        return self.c_rt_cgs

    @property
    def light_speed_code(self) -> float:
        """RT light speed in code units, normalized by hydro velocity scale."""
        return self.c_rt_cgs / self.V_cgs

    def scale(self, dim: str) -> float:
        scales = {
            "length": self.L_cgs,
            "mass": self.M_cgs,
            "time": self.T_cgs,
            "velocity": self.V_cgs,
            "density": self.rho_cgs,
            "pressure": self.P_cgs,
            "energy_density": self.Eden_cgs,
            "radiation_energy_density": self.RadEnergy_cgs,
            "photon_number": self.PhotonNumber_cgs,
            "photon_density": self.PhotonSurfaceDensity_cgs,
            "photon_surface_density": self.PhotonSurfaceDensity_cgs,
            "photon_number_density": self.PhotonNumberDensity_cgs,
            "photon_flux": self.PhotonFlux_cgs,
            "photon_rate": self.PhotonRate_cgs,
            "temperature": self.Temp_cgs,
            "radiation_flux": self.RadFlux_cgs,
            "light_speed": self.light_speed_cgs,
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

        c_rt_cfg = code_units_cfg.get("light_speed", "3e10 cm/s")
        c_rt_cgs = _parse_base_scale(c_rt_cfg, "velocity", unit_parser)

        return cls(
            L_cgs=L_cgs,
            M_cgs=M_cgs,
            V_cgs=V_cgs,
            c_rt_cgs=c_rt_cgs,
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